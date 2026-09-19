"""Encode endpoint-inclusive MCP recordings and verify every scored capture metric."""
import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ffmpeg', required=True)
    args = parser.parse_args()
    root = Path(__file__).parent
    pairs = []
    for scene in ('panda', 'allegro', 'hug'):
        evidence_path = root / f'data/{scene}-mcp-evidence.json'
        evidence = json.loads(evidence_path.read_text())
        capture = evidence['calls'][1]['result']['result']['metrics']
        verifier_path = root / f'data/confirmation/trials/{scene}-live-0/verification/metrics.json'
        verifier = json.loads(verifier_path.read_text())
        keys = list(verifier['thresholds']) + ['thresholds', 'config', 'frames', 'expected_frames', 'finite', 'sample_count', 'success']
        values = {key: {'capture': capture[key], 'verifier': verifier[key], 'equal': capture[key] == verifier[key]} for key in keys}
        if not all(value['equal'] for value in values.values()):
            raise ValueError(f'{scene}: capture does not match the final verifier')
        frames = root / f'assets/{scene}-mcp-recording'
        if len(list(frames.glob('frame-*.png'))) != 31:
            raise ValueError(f'{scene}: expected all 31 frames')
        subprocess.run([args.ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-framerate', '10', '-i', str(frames / 'frame-%06d.png'), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(root / f'assets/{scene}-mcp.mp4')], check=True)
        pairs.append({'scenario': scene, 'variant': 0, 'capture_evidence': str(evidence_path.relative_to(root)), 'capture_sha256': hashlib.sha256(evidence_path.read_bytes()).hexdigest(), 'fresh_verifier': str(verifier_path.relative_to(root)), 'all_scored_metrics_exactly_equal': True, 'comparisons': values, 'frames': 31, 'simulation_span_s': [0, 3], 'presentation_fps': 10, 'playback_duration_s': 3.1, 'duration_note': '31 endpoint-inclusive frames sampled every 0.1 s; final frame held for 0.1 s'})
    (root / 'data/capture-parity.json').write_text(json.dumps({'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'outside_timed_agent_trials': True, 'comparisons': pairs}, indent=2) + '\n')
    print(json.dumps({'encoded_videos': 3, 'capture_parity': 'all scored metrics exactly equal'}))


if __name__ == '__main__':
    main()
