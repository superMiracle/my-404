import asyncio
import base64
import time
import typing
import urllib.parse

import aiohttp
import bittensor as bt
import pyspz
from aiohttp import ClientTimeout
from aiohttp.helpers import sentinel
from common.miner_license_consent_declaration import MINER_LICENSE_CONSENT_DECLARATION
from common.protocol import PullTask, SubmitResults

from miner import ValidatorSelector


NETWORK_DELAY_TIME_BUFFER = 60
FAILED_VALIDATOR_DELAY = 300


async def validate_result(validation_endpoint: str, prompt: str, data: str) -> float | None:
    validate_url = urllib.parse.urljoin(validation_endpoint, "/validate_txt_to_3d_ply/")
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(validate_url, json={"prompt": prompt, "data": data, "compression": 2}) as response:
                if response.status == 200:
                    results = await response.json()
                    return float(results["score"])
        except Exception as e:
            bt.logging.error(f"Validation error: {e}")
    return None


async def worker_routine(
    endpoint: str, wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    bt.logging.info(f"Worker ({endpoint}) started")
    generate_url = urllib.parse.urljoin(endpoint, "/generate/")

    # Import config here to get validation settings
    from miner.config import read_config
    config = read_config()
    validation_endpoint = config.validation.endpoint
    quality_threshold = config.validation.quality_threshold

    while True:
        validator_uid = validator_selector.get_next_validator_to_query()
        if validator_uid is None:
            await asyncio.sleep(10.0)
            continue

        # Setting cooldown to prevent selecting the same validator for concurrent task.
        validator_selector.set_cooldown(validator_uid, int(time.time()) + 300)

        async with bt.dendrite(wallet=wallet) as dendrite:
            pull = await _pull_task(dendrite, metagraph, validator_uid)
            if pull.dendrite.status_code != 200:
                bt.logging.warning(
                    f"Failed to get task from [{validator_uid}]. Reason: {pull.dendrite.status_message}."
                )
                validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
                continue

        if pull.task is None:
            if pull.cooldown_until == 0:
                bt.logging.warning(f"Failed to get task from [{validator_uid}]. Reason: Unknown.")
                validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            else:
                cooldown_left = max(0, int(pull.cooldown_until - time.time()))
                bt.logging.debug(
                    f"Miner is on cooldown for the next: {cooldown_left} sec. "
                    f"Total cooldown violations: {pull.cooldown_violations}"
                )
                validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
            continue

        bt.logging.debug(f"Task received. Prompt: {pull.task.prompt}.")

        results = await _generate(generate_url, pull.task.prompt) or b""

        # Validate result before submitting
        if results:
            encoded_results = base64.b64encode(results).decode(encoding="utf-8")
            
            asyncio.create_task(_submit_and_log(wallet, metagraph, validator_uid, pull, results, validator_selector))
            
        else:
            bt.logging.info("No results generated, submitting empty results.")
            asyncio.create_task(_submit_and_log(wallet, metagraph, validator_uid, pull, b'', validator_selector))


async def _complete_one_task(
    generate_url: str, wallet: bt.wallet, metagraph: bt.metagraph, validator_selector: ValidatorSelector
) -> None:
    validator_uid = validator_selector.get_next_validator_to_query()
    if validator_uid is None:
        await asyncio.sleep(10.0)
        return

    # Setting cooldown to prevent selecting the same validator for concurrent task.
    validator_selector.set_cooldown(validator_uid, int(time.time()) + 300)

    async with bt.dendrite(wallet=wallet) as dendrite:
        pull = await _pull_task(dendrite, metagraph, validator_uid)
        if pull.dendrite.status_code != 200:
            bt.logging.warning(
                f"Failed to get task from [{validator_uid}]. Reason: {pull.dendrite.status_message}."
            )
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            return

    if pull.task is None:
        if pull.cooldown_until == 0:
            bt.logging.warning(f"Failed to get task from [{validator_uid}]. Reason: Unknown.")
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
        else:
            cooldown_left = max(0, int(pull.cooldown_until - time.time()))
            bt.logging.debug(
                f"Miner is on cooldown for the next: {cooldown_left} sec. "
                f"Total cooldown violations: {pull.cooldown_violations}"
            )
            validator_selector.set_cooldown(validator_uid, pull.cooldown_until)
        return

    bt.logging.debug(f"Task received. Prompt: {pull.task.prompt}.")

    results = await _generate(generate_url, pull.task.prompt) or b""

    async with bt.dendrite(wallet=wallet) as dendrite:
        submit = await _submit_results(wallet, dendrite, metagraph, validator_uid, pull, results)
        if submit.feedback is None:
            bt.logging.warning(
                f"Failed to submit results to [{metagraph.hotkeys[validator_uid]}]. "
                f"Reason: {submit.dendrite.status_message}."
            )
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            return

    _log_feedback(validator_uid, submit)

    validator_selector.set_cooldown(validator_uid, submit.cooldown_until)


async def _pull_task(dendrite: bt.dendrite, metagraph: bt.metagraph, validator_uid: int) -> PullTask:
    synapse = PullTask()
    response = typing.cast(
        PullTask,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid], synapse=synapse, deserialize=False, timeout=12.0
        ),
    )
    return response


async def _submit_results(
    wallet: bt.wallet,
    dendrite: bt.dendrite,
    metagraph: bt.metagraph,
    validator_uid: int,
    pull: PullTask,
    results: bytes,
) -> SubmitResults:
    from miner.config import read_config
    config = read_config()
    validation_endpoint = config.validation.endpoint
    quality_threshold = config.validation.quality_threshold
    
    submit_time = time.time_ns()
    prompt = pull.task.prompt if pull.task is not None else None
    message = (
        f"{MINER_LICENSE_CONSENT_DECLARATION}"
        f"{submit_time}{prompt}{metagraph.hotkeys[validator_uid]}{wallet.hotkey.ss58_address}"
    )
    signature = base64.b64encode(dendrite.keypair.sign(message)).decode(encoding="utf-8")
    if results:
        compressed_results = base64.b64encode(pyspz.compress(results, workers=-1)).decode(encoding="utf-8")
        validation_score = await validate_result(validation_endpoint, pull.task.prompt, compressed_results)
        
        if validation_score is not None and validation_score >= quality_threshold:
            bt.logging.info(f"Validation passed (score: {validation_score}), submitting result.")
        else:
            bt.logging.info(f"Validation failed or below threshold (score: {validation_score}, threshold: {quality_threshold}), submitting empty results.")
            compressed_results = ""  # Skipping task not to be penalized (same could be done for low quality results)
    else:
        compressed_results = ""  # Skipping task not to be penalized (same could be done for low quality results)
        
    synapse = SubmitResults(task=pull.task, results=compressed_results, submit_time=submit_time, signature=signature)
    response = typing.cast(
        SubmitResults,
        await dendrite.call(
            target_axon=metagraph.axons[validator_uid],
            synapse=synapse,
            deserialize=False,
            timeout=300.0,
        ),
    )
    return response


def _log_feedback(validator_uid: int, submit: SubmitResults) -> None:
    feedback = submit.feedback
    if feedback is None:
        return
    score = "failed" if feedback.validation_failed else feedback.task_fidelity_score
    bt.logging.debug(f"Feedback received from [{validator_uid}]. Prompt: {submit.task.prompt}. Score: {score}")
    bt.logging.debug(
        f"Average score: {feedback.average_fidelity_score}. "
        f"Accepted results (last 4h): {feedback.generations_within_the_window}. "
        f"Reward: {feedback.current_miner_reward}."
    )


async def _submit_and_log(wallet, metagraph, validator_uid, pull, results, validator_selector):
    async with bt.dendrite(wallet=wallet) as dendrite:
        submit = await _submit_results(wallet, dendrite, metagraph, validator_uid, pull, results)
        if submit.feedback is None:
            bt.logging.warning(
                f"Failed to submit results to [{metagraph.hotkeys[validator_uid]}]. "
                f"Reason: {submit.dendrite.status_message}."
            )
            validator_selector.set_cooldown(validator_uid, int(time.time()) + FAILED_VALIDATOR_DELAY)
            return
    _log_feedback(validator_uid, submit)
    validator_selector.set_cooldown(validator_uid, submit.cooldown_until)


async def _generate(generate_url: str, prompt: str, timeout: float | None = None) -> bytes | None:  # noqa: ASYNC109
    bt.logging.debug(f"Generating for prompt: {prompt} with timeout {timeout} seconds")

    client_timeout = ClientTimeout(total=timeout) if timeout is not None else sentinel
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        try:
            async with session.post(generate_url, data={"prompt": prompt}) as response:
                if response.status == 200:
                    results = await response.read()
                    bt.logging.debug(f"Generation completed. Size: {len(results)}")
                    return results
                else:
                    bt.logging.error(f"Generation failed with code: {response.status}")
        except aiohttp.ClientConnectorError:
            bt.logging.error(f"Failed to connect to the endpoint. The endpoint might be inaccessible: {generate_url}.")
        except TimeoutError:
            bt.logging.error(f"The request to the endpoint timed out: {generate_url}")
        except aiohttp.ClientError as e:
            bt.logging.error(f"An unexpected client error occurred: {e} ({generate_url})")
        except Exception as e:
            bt.logging.error(f"An unexpected error occurred: {e} ({generate_url})")
