#!/bin/bash

# This script automates creating a PR for merging to upstream based on
# the PR created for innersource.
# This script is to be run from the innersource PR branch.

# Step 0: fetch latest changes from innersource and upstream
# If remote for upstream does not exist, add it
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
# fetch from innersource(origin) and upstream
git fetch origin
git fetch upstream
# Checkout origin/main if origin/main is merged into this PR branch
if [ `git merge-base --is-ancestor origin/main HEAD` -ne 0 ]
then
    echo "Please merge origin/main first, resolve merge conflicts if any and rerun this script."
    exit 1
fi

# Step 1: Create a squashed single commit from current PR
curr_branch=`git rev-parse --abbrev-ref HEAD`
echo "Innersource PR branch name: ${curr_branch}"
# Create a temp branch for squashing, if it already exists, delete it first.
if [ `git rev-parse --verify temp_squashed_pr 2>/dev/null` ]
then
   git branch -D temp_squashed_pr
fi
git checkout -b temp_squashed_pr origin/main
# Create a single squashed commit on top of innersource main
git merge --squash ${curr_branch} && git commit --no-edit
single_commit=`git rev-parse temp_squashed_pr`

# Step 2: Create a new PR branch for upstreaming with the single commit
# branch name: <current_branch>_for_upstream
new_branch="${curr_branch}_for_upstream"
echo "Public PR branch name: ${new_branch}"
if [ `git rev-parse --verify ${new_branch} 2>/dev/null` ]
then
   git branch -D ${new_branch}
fi
# create a new branch for upstreaming
git checkout -b ${new_branch} upstream/main
# cherry-pick the single commit and ask user to update commit message.
git cherry-pick ${single_commit} && git commit --amend
