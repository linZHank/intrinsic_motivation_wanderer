# Workflow

1. On desktop
- Specify model loading directory around **line 34** in `play.py` to `<this repository>/model_dir/<version>/<latest date>`
- Git add, commit, push.

2. On TX2
- Run `play.py`
- `ls /ssd/mecanum_experience` to find out saved experience data directory.

3. Back to desktop
- `scp` latest collected experience
- Open `learn.py`, specify model loading directory(around line 29), data loading directory(around line 37) and previous data loading directory(around line 67).
- Run `learn.py`
- Repeat 1. and 2.
