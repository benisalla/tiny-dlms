"""Shared helpers for the demo videos (scripts/car_drive.py, rhoa_demo.py, maze_play.py)."""
import os
import subprocess


def load_font(size):
    """A font that has Greek-style letters (ē) when one is installed, else Pillow's built-in one."""
    from PIL import ImageFont
    for path in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def save_video(frames, paths, fps=25, gif_size=None, gif_every=1, gif_max=None):
    """
        PIL images -> each path: .mp4 (ffmpeg, H.264) or .gif (Pillow).
        GIFs are for READMEs and pages, so they can be lighter: resized to gif_size, one frame in gif_every,
        at most gif_max frames.
    """
    for path in paths:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if path.endswith(".gif"):
            small = frames[::gif_every][:gif_max]
            small = [f.resize(gif_size) for f in small] if gif_size else small
            small[0].save(path, save_all=True, append_images=small[1:], duration=int(1000 * gif_every / fps),
                          loop=0, optimize=True)
            n = len(small)
        else:
            w, h = frames[0].size
            cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path]
            subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)
            n = len(frames)
        print(f"saved {n} frames to {path}")
