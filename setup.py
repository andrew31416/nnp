import os

# move pre-push into .git/hooks for use when pushing to repo


hooks = ['pre-push']

for _hook in hooks:
    os.symlink(src='../../util/{}'.format(_hook),dst='.git/hooks/{}'.format(_hook))
