import asyncio
import aiohttp
import base64
import pyspz
import sys
import os

# Demo config
ENDPOINT = "http://localhost:8093/generate/"
PROMPT = "shimmering metallic ore samples"  # Change as needed
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
            print(f"Error during generation: {e}")
    return b""

def compress_mesh(mesh_bytes: bytes) -> bytes:
    try:
        compressed = pyspz.compress(mesh_bytes, workers=-1)
        print(f"Compression succeeded. Compressed size: {len(compressed)} bytes.")
        return compressed
    except Exception as e:
        print(f"Compression failed: {e}")
        return b""

def decompress_mesh(compressed_bytes: bytes) -> bytes:
    try:
        decompressed = pyspz.decompress(compressed_bytes)
        print(f"Decompression succeeded. Decompressed size: {len(decompressed)} bytes.")
        return decompressed
    except Exception as e:
        print(f"Decompression failed: {e}")
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
        decompressed_bytes = decompress_mesh(compressed_bytes)
        if decompressed_bytes != mesh_bytes:
            print("Warning: Decompressed mesh does not match original!")
        else:
            print("Round-trip compression/decompression succeeded.")

    asyncio.run(run())

if __name__ == "__main__":
    main() 