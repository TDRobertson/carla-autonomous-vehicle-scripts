import subprocess
import sys
import os

SCRIPTS = [
    ("sensor_fusion_testing/scene.py", "=== Running scene.py ===", "80x24+0+0"),
    ("sensor_fusion_testing/fpv_ghost.py", "=== Running fpv_ghost.py ===", "80x24+960+0"),
    ("sensor_fusion_testing/sync.py", "=== Running sync.py ===", "80x24+0+540"),
]

def run_in_new_console(script, header, geometry):
    if sys.platform == "win32":
        # Windows: print header, activate venv, run script
        subprocess.Popen([
            'start', 'cmd', '/k', f'echo {header} && venv\\Scripts\\activate && python {script}'
        ], shell=True)
    else:
        # Linux/WSL: use gnome-terminal with geometry and header
        subprocess.Popen([
            'gnome-terminal', '--geometry', geometry, '--', 'bash', '-c',
            f"echo '{header}'; source venv/bin/activate && python {script}; exec bash"
        ])

if __name__ == "__main__":
    for script, header, geometry in SCRIPTS:
        run_in_new_console(script, header, geometry)
