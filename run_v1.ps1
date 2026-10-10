# Reproduce every v1 run (fp32, greedy, 12 threads, 96 new tokens). ~2 h on an 8-core laptop CPU.
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONIOENCODING = 'utf-8'
$py = '.\.venv\Scripts\python.exe'
$common = @('--threads', '12', '--max-new-tokens', '96')

& $py run_dialogue.py --suite suite\v1\handwritten.jsonl     --out artifacts\v1\main     @common
& $py run_dialogue.py --suite suite\v1\escalating_harm.jsonl --out artifacts\v1\harm     @common
& $py run_dialogue.py --suite suite\v1\controls.jsonl        --out artifacts\v1\controls @common
# Precision / determinism spot-check on the first 4 harm items.
& $py run_dialogue.py --suite suite\v1\escalating_harm.jsonl --out artifacts\v1\repeat_fp32 --limit 4 @common
& $py run_dialogue.py --suite suite\v1\escalating_harm.jsonl --out artifacts\v1\repeat_bf16 --limit 4 --dtype bfloat16 @common
