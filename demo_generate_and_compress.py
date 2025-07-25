import asyncio
import aiohttp
import base64
import pyspz
import sys
import os

# Demo config
ENDPOINT = "http://localhost:8093/generate/"
PROMPT = "orange sturdy hockey stick"  # Change as needed
MESH_OUTPUT = "demo_output.ply"
COMPRESSED_OUTPUT = "demo_output.ply.spz"

async def generate_3d(prompt: str, endpoint: str = ENDPOINT) -> bytes:
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(endpoint, data={"prompt": prompt}) as response:
                if response.status == 200:
                    mesh_bytes = await response.read()
                    print(f"Generation succeeded. Mesh size: {len(mesh_bytes)} bytes.")
                    return mesh_bytes
                else:
                    print(f"Generation failed with code: {response.status}")
        except Exception as e:
            print(f"Error during generation: ")
    return b""

async def validate_result(validation_endpoint: str, prompt: str, data: str) -> float | None:
    validate_url = validation_endpoint.rstrip("/") + "/validate_txt_to_3d_ply/"
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(validate_url, json={"prompt": prompt, "data": data, "compression": 2}) as response:
                if response.status == 200:
                    results = await response.json()
                    return float(results["score"])
                else:
                    print(f"Validation failed with code: {response.status}")
        except Exception as e:
            print(f"Validation error: ")
    return None

def compress_mesh(mesh_bytes: bytes) -> bytes:
    try:
        compressed = pyspz.compress(mesh_bytes, workers=-1)
        print(f"Compression succeeded. Compressed size: {len(compressed)} bytes.")
        return compressed
    except Exception as e:
        print(f"Compression failed: ")
        return b""

def decompress_mesh(compressed_bytes: bytes) -> bytes:
    try:
        decompressed = pyspz.decompress(compressed_bytes)
        print(f"Decompression succeeded. Decompressed size: {len(decompressed)} bytes.")
        return decompressed
    except Exception as e:
        print(f"Decompression failed: ")
        return b""

def save_file(path: str, data: bytes):
    with open(path, "wb") as f:
        f.write(data)
    print(f"Saved file: {path} ({len(data)} bytes)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Demo 3D generation and compression.")
    parser.add_argument("--prompt", type=str, default=PROMPT, help="Prompt for 3D generation")
    parser.add_argument("--endpoint", type=str, default=ENDPOINT, help="Generation endpoint URL")
    parser.add_argument("--mesh", type=str, default=MESH_OUTPUT, help="Output mesh file path")
    parser.add_argument("--compressed", type=str, default=COMPRESSED_OUTPUT, help="Output compressed file path")
    parser.add_argument("--validation-endpoint", type=str, default="http://127.0.0.1:8094", help="Validation endpoint URL")
    parser.add_argument("--validation-threshold", type=float, default=0.6, help="Validation score threshold")
    args = parser.parse_args()

    async def run():
        mesh_bytes = await generate_3d(args.prompt, args.endpoint)
        if not mesh_bytes:
            print("No mesh generated. Exiting.")
            return
        save_file(args.mesh, mesh_bytes)
        compressed_bytes = compress_mesh(mesh_bytes)
        if not compressed_bytes:
            print("Compression failed. Exiting.")
            return
        save_file(args.compressed, compressed_bytes)
        # Optional: verify decompression
        # decompressed_bytes = decompress_mesh(compressed_bytes)
        # if decompressed_bytes != mesh_bytes:
        #     print("Warning: Decompressed mesh does not match original!")
        # else:
        #     print("Round-trip compression/decompression succeeded.")
        # Validation check
        encoded_data = base64.b64encode(compressed_bytes).decode(encoding="utf-8")
        score = await validate_result(args.validation_endpoint, args.prompt, encoded_data)
        if score is not None:
            print(f"Validation score: {score}")
            if score < args.validation_threshold:
                print(f"Warning: Validation score {score} is below threshold {args.validation_threshold}!")
            else:
                print("Validation passed.")
        else:
            print("Validation failed or no score returned.")

    asyncio.run(run())

if __name__ == "__main__":
    main() 