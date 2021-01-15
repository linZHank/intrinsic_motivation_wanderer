# Workflow

1. On desktop
- Specify model loading directory around **line 34** in `play.py` to `<this repository>/model_dir/<version>/<latest date>`
- Git add, commit, push.

2. On TX2
- Git pull
- **IMPORTANT** Unplug Ethernet cable
- Run `play.py`
- `ls /ssd/mecanum_experience` to find out saved experience data directory.
- Plug Ethernet cable

3. Back to desktop
- `scp` latest collected experience
- Open `learn.py`, specify model loading directory(around line 29), data loading directory(around line 37) and previous data loading directory(around line 67).
- Run `learn.py`
- Repeat step 1 and 2.
