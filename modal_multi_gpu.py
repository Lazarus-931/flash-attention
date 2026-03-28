import modal

app = modal.App("fa4-fused-kv-bench")

base_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git")
    .pip_install("wheel", "setuptools", "numpy")
    .pip_install("torch==2.6.0", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("einops", "ninja", "packaging", "typing_extensions")
    .run_commands(
        "pip install 'apache-tvm-ffi>=0.1.5,<0.2'",
        "pip install torch-c-dlpack-ext --no-binary torch-c-dlpack-ext --no-build-isolation",
        "pip install 'nvidia-cutlass-dsl>=4.4.2' 'quack-kernels>=0.3.3'",
        "git clone -b turboquant-dequant https://github.com/Lazarus-931/flash-attention.git /root/fa3",
        "cd /root/fa3/flash_attn/cute && pip install -e '.[dev]' --no-deps",
    )
)


def run_bench():
    import subprocess
    r = subprocess.run(
        ["python", "benchmarks/bench_fa4_dequant_kernel.py"],
        capture_output=True, text=True, cwd="/root/fa3",
    )
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-2000:])
    return r.stdout


@app.function(image=base_image, gpu="A100", timeout=1800)
def bench_a100():
    return run_bench()


@app.function(image=base_image, gpu="H100", timeout=1800)
def bench_h100():
    return run_bench()


@app.local_entrypoint()
def main():
    jobs = [
        ("A100", bench_a100.spawn()),
        ("H100", bench_h100.spawn()),
    ]
    for label, handle in jobs:
        print(f"\n{'='*70}")
        print(f"  {label}")
        print(f"{'='*70}")
        try:
            print(handle.get())
        except Exception as e:
            print(f"  FAILED: {e}")
