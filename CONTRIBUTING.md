# How we work on this repo

Three of us share this repo. We commit straight to `main` — no pull requests,
no approvals. The rules below are what keep that from turning into a mess.

## One-time setup

```bash
git clone https://github.com/Benedict-vs/bomberman_RL.git
cd bomberman_RL
git config pull.rebase true      # keeps history readable instead of full of merge commits
```

Tell git who you are, if you never have:

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

## The everyday loop

```bash
git pull                    # 1. before you start: get everyone else's work
                            # 2. ...edit files, train, test...
git status                  # 3. look at what you changed
git add -A                  # 4. stage it
git commit -m "Add distance-to-coin reward shaping"
git push                    # 5. share it
```

That's it. Do this several times a day, not once a week.

## Important

**1. Pull before you start, push when you stop.**
Unpushed work is invisible to everyone else and gets harder to merge every hour
it sits on your machine. Push even if it isn't finished — small commits are fine.

**2. Read `git status` before every `git add -A`.**
`-A` stages *everything*, including stray model checkpoints from a training run.
Logs, replays, results, screenshots and `__pycache__` are already ignored, so it's
mostly model files to watch for. If you see something you didn't mean to commit,
stage the specific paths instead:

```bash
git add agent_code/dqn_agent/callbacks.py agent_code/dqn_agent/train.py
```

**3. Don't commit every training checkpoint.**
Binary model files can't be merged by git — if two of us commit the same one, one
version simply wins. Commit a model when it's a milestone worth sharing, not after
every run.

## Updating framework later
```bash
git fetch upstream && git merge upstream/master
```
No `--allow-unrelated-histories` needed

## When `git push` is rejected

It means someone pushed while you were working. Normal, not a problem:

```bash
git pull        # pulls their work and replays yours on top
git push
```

If that reports a **conflict**, git has marked the clashing lines inside the file with
`<<<<<<<` / `>>>>>>>`. Edit the file so it reads the way it should, delete the markers,
then:

```bash
git add <the-file>
git rebase --continue
git push
```

If it looks scary, stop and ask in the group chat before typing anything else. Nothing
is lost at this point — it's always recoverable.

## Branches (only when you need them)

For a risky experiment you don't want on `main` yet, make a branch named after yourself:

```bash
git switch -c <name>/try-cnn-features
# ...work, commit, push...
git push -u origin <name>/try-cnn-features
```

Bring it back when it works:

```bash
git switch main
git pull
git merge benni/try-cnn-features
git push
git branch -d benni/try-cnn-features
```

Keep these short-lived — days, not weeks. A branch that sits open while `main` moves on
is where the painful conflicts come from.

## Useful

```bash
git log --oneline -10       # recent commits
git diff                    # what you changed but haven't staged
git switch -c tmp           # oops, I committed on main by accident:
                            #   this carries your commits onto a new branch
```

## Running the game

```bash
python main.py play --my-agent <your_agent>
python main.py play --agents <your_agent> rule_based_agent --train 1 --no-gui
python main.py play --help
```
