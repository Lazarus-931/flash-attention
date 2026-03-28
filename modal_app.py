import modal

app = modal.App("fa4-turboquant-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("wheel", "setuptools", "numpy")
    .pip_install(
        "torch==2.6.0",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("einops", "ninja", "packaging", "typing_extensions")
    .run_commands(
        "pip install 'apache-tvm-ffi>=0.1.5,<0.2'",
        "pip install torch-c-dlpack-ext --no-binary torch-c-dlpack-ext --no-build-isolation",
        "pip install 'nvidia-cutlass-dsl>=4.4.2' 'quack-kernels>=0.3.3'",
    )
    # Separate step to avoid cache of old clone
    .run_commands(
        "git clone -b turboquant-dequant https://github.com/Lazarus-931/flash-attention.git /root/flash-attention-v18",
        "cd /root/flash-attention-v18/flash_attn/cute && pip install -e '.[dev]' --no-deps",
    )
)


@app.function(image=image, gpu="H100", timeout=1800)
def benchmark():
    import subprocess
    result = subprocess.run(
        ["python", "benchmarks/bench_fa4_dequant_kernel.py"],
        capture_output=True,
        text=True,
        cwd="/root/flash-attention-v18",
    )
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr[-5000:])
    return result.stdout


@app.local_entrypoint()
def main():
    output = benchmark.remote()
    print(output)
