# IMEX development flow
## Intro
IMEX currently has two repos. One is the public IMEX repo and other is an innersource IMEX repo. Recommended development model for internal developers is

Step 0: Do initial setup – see instructions below.

Step 1: Develop features using a branch and then create a PR against innersource IMEX repo’s “main” branch. Code review and CI checks will be done using innersource repo.

Step 2: Run “prepare_upstream.sh” script to create a new branch based on public repo “main” and adds a single squashed commit from PR. You would be prompted to edit the squashed commit message. If you PR branch was named “foo” the new branch will be auto named as “foo_for_upstream”
```
$> ./dev_scripts/prepare_upstream.sh
```

Step 3: Push the new branch to public repo “main”.
```
$> git push origin foo_for_upstream main
```

Step 4: Delete the new branch.

## Initial setup
### 1.	Setup innersource and checkout innersource IMEX
Step 1: download dt
```
$> curl -fL https://goto.intel.com/getdt | sh
```
Step 2: make downloaded dt executable
```
$> chmod +x dt
```
Step 3: install a local copy of dt
```
$> ./dt install
```
Step 4: remove downloaded dt
```
$> rm dt
```
Step 5: setup with installed dt
```
$> dt setup
```
Follow instructions on screen. Make sure during dt setup to generate new token. And select netrc. (should be default) This will help setup netrc file with proper credentials

Step 6: clone imex innersource repo
```
$> git clone https://github.com/intel-innersource/frameworks.ai.mlir.mlir-extensions.git
```

### 2.	(Optional) Create personal innersource IMEX fork (Skip this part if you prefer working directly on innersource IMEX branches)

Step 1: Click “Fork” button on https://github.com/intel-innersource/frameworks.ai.mlir.mlir-extensions

Step 2: Select your account in the “owner” pull down list and click “Create fork”

At this point you would see a personal fork on github. Something like https://github.com/silee2/frameworks.ai.mlir.mlir-extensions

Step 3: add a new remote
```
$> git remote add silee2 https://github.com/silee2/frameworks.ai.mlir.mlir-extensions
```

And you would see a new “remote” is added for the cloned repo

Step 4: check git remote
```
$> git remote
origin
silee2
```

