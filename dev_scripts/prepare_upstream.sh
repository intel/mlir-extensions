#!/bin/bash

# This scripts automates created a PR for upstream base on
# PR created for innersource

# Step 0: fetch latest changes from innersource and upstream
if [ `git config remote.upstream.url 2> /dev/null` ]
then
    if [ `git config remote.upstream.url 2> /dev/null` != "https://github.com/intel/mlir-extensions.git" ]
    then
        echo "Remote upstream already exists and does not match public IMEX"
        exit 1
    fi
else
   git remote add upstream https://github.com/intel/mlir-extensions.git
fi
git fetch origin
git fetch upstream

# Step 1: Create a squashed single commit from current PR
curr_branch=`git rev-parse --abbrev-ref HEAD`
echo "Innersource PR branch name: ${curr_branch}"
if [ `git rev-parse --verify temp_squashed_pr 2>/dev/null` ]
then
   git branch -D temp_squashed_pr
fi
git checkout -b temp_squashed_pr origin/main
git merge --squash ${curr_branch} && git commit --no-edit
single_commit=`git rev-parse temp_squashed_pr`

# Step 2: Create a new PR branch for upstreaming with the single commit
new_branch="${curr_branch}_for_upstream"
echo "Public PR branch name: ${new_branch}"
if [ `git rev-parse --verify ${new_branch} 2>/dev/null` ]
then
   git branch -D ${new_branch}
fi
git checkout -b ${new_branch} upstream/main
git cherry-pick ${single_commit}

# Step 3: Update commit message
git commit --amend
