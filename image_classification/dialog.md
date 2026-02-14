# Development Session Log

This document captures the conversation between a developer and Claude Code while setting up this repository and building out the project.

---

## Setting Up the Repository

### Creating the project directory and initializing git

```bash
pwd
# /Users/johnzastrow

mkdir Github
cd Github
mkdir ms
cd ms

ls -lht
# -rw-r--r--@ 1 johnzastrow  staff   213K Feb 14 09:41 CSYE7380_W05_Agentic_AI_for_Image_Classification (1).ipynb
```

### Installing git via Homebrew

```bash
brew install git
# ==> Installing git dependency: libiconv
# ==> Pouring libiconv--1.18.arm64_tahoe.bottle.tar.gz
# ==> Pouring git--2.53.0.arm64_tahoe.bottle.tar.gz
# 🍺  /opt/homebrew/Cellar/git/2.53.0: 1,741 files, 62.0MB
```

### Renaming the notebook and initializing the repo

```bash
mv CSYE7380_W05_Agentic_AI_for_Image_Classification\ \(1\).ipynb image_class.ipynb

git init
# Initialized empty Git repository in /Users/johnzastrow/Github/ms/.git/

git add .
git commit -m "Starting commit"
# [main (root-commit) 8745398] Starting commit
#  1 file changed, 1 insertion(+)
#  create mode 100644 image_class.ipynb
```

### Configuring git and GitHub CLI

```bash
git config --global user.name "JohnZastrow"
git config --global user.email "br8kwall@gmail.com"
git config --global core.editor nano

gh auth login
# ? Where do you use GitHub? GitHub.com
# ? What is your preferred protocol for Git operations on this host? SSH
# ? Generate a new SSH key to add to your GitHub account? Yes
# ? Enter a passphrase for your new SSH key (Optional): ********
# ? Title for your SSH key: GitHub CLI
# ? How would you like to authenticate GitHub CLI? Login with a web browser
# ✓ Authentication complete.
# ✓ Uploaded the SSH key to your GitHub account: /Users/johnzastrow/.ssh/id_ed25519.pub
# ✓ Logged in as johnzastrow
```

### Creating the GitHub repository

```bash
gh repo create
# ? What would you like to do? Push an existing local repository to github.com
# ? Path to local repository .
# ? Repository name ms
# ? Repository owner johnzastrow
# ? Description For Hynes' stuff
# ? Visibility Public
# ✓ Created repository johnzastrow/ms on github.com
#   https://github.com/johnzastrow/ms
# ? Add a remote? Yes
# ? What should the new remote be called? origin
# ✓ Added remote git@github.com:johnzastrow/ms.git
# ✓ Pushed commits to git@github.com:johnzastrow/ms.git
```

---

## Starting Claude Code

```
claude
# ╭─── Claude Code v2.1.42 ──────────────────────────────────────╮
# │                           │ Tips for getting started          │
# │      Welcome back JZ!     │ Run /init to create a CLAUDE.md   │
# │          ▐▛███▜▌          │                                   │
# │         ▝▜█████▛▘         │                                   │
# │           ▘▘ ▝▝           │                                   │
# │   Opus 4.6 · Claude Max   │                                   │
# │        ~/Github/ms        │                                   │
# ╰───────────────────────────────────────────────────────────────╯
```

---

## Exploring the Codebase

**User:** have a look around

**Claude:** Here's what's in this repo:

**Single file:** `image_class.ipynb` -- a Jupyter notebook titled "Agentic AI for Image Classification".

It implements a multi-agent system using Kafka as a message broker to compare two image classifiers on CIFAR-10. Here's the architecture:

1. **CNN Agent** -- a custom CNN model that subscribes to a Kafka topic, classifies incoming images, and publishes results
2. **ResNet Agent** -- a ResNet-style model (with residual blocks) that does the same
3. **Judge Agent** -- sends 500 random CIFAR-10 test images to both agents via Kafka, collects their responses, and picks the higher-confidence prediction as the "brokered" result

**Workflow (10 steps):**
- Steps 1-2: Install deps, download/start Kafka+Zookeeper, load CIFAR-10
- Steps 3-4: Build & train CNN and ResNet models (with early stopping, LR scheduling)
- Steps 5-6: Create CNN and ResNet agents as Kafka consumers/producers running in background threads
- Steps 7-8: Judge agent publishes requests and collects/compares responses; orchestration via threading
- Steps 9-10: Kafka restart (cleanup), run the full system, and visualize accuracy with a bar chart

The notebook was run in Google Colab and all cells have outputs. It appears to be a class assignment (comments reference previous weekly assignments like `W03_5_Publish_Subscribe.ipynb` and `W04_CNN.ipynb`).

---

## Can This Run Without a GPU?

**User:** will this run on my own jupyterhub without a GPU?

**Claude:** **Short answer: Yes, it will run without a GPU, but with some caveats.**

**What will work fine:**
- TensorFlow/Keras runs on CPU by default -- no code changes needed
- The CIFAR-10 dataset is small (32x32 images) and downloads automatically
- The models are relatively small (not deep ResNets)

**Pain points without a GPU:**
- **Training will be slow.** On CPU, expect each epoch to take several minutes instead of seconds. The CNN trains up to 50 epochs (with early stopping), and the ResNet does the same. You could be waiting 30-60+ minutes total for training.
- **Inference for 500 images** will also be slower, but manageable.

**The real blocker is Kafka, not the GPU.** The notebook:
- Downloads and runs Apache Kafka + Zookeeper locally via shell commands (`!curl`, `!./kafka_2.13-3.5.0/bin/...`)
- Uses `!` shell commands written for a Linux environment (Colab)
- Kills processes with `xargs -r kill -9` (the `-r` flag is a GNU/Linux extension -- on macOS it won't work the same way)
- Expects to write to `/tmp/kafka-logs` and `/tmp/zookeeper`

**To run on your local JupyterHub you'd need to:**
1. Have Java installed (Kafka requires it)
2. Adjust or pre-install Kafka rather than downloading it in-cell
3. Fix macOS-incompatible shell commands (like `xargs -r`)
4. Potentially adjust paths and process management

**Tip to speed up training without a GPU:** reduce epochs (e.g., 10-15 instead of 50 -- early stopping will likely kick in anyway) or reduce the training set size.

---

## Creating the VPS Deployment Guide

**User:** create a new file called RUN_ON_VPS.md with detailed instructions for setting up jupyterhub on an Ubuntu 24.04 host and then configuring it to run this notebook first on a VPS with no GPU, then one with an RTX4000 and 16GB of memory

**Claude:** Created `RUN_ON_VPS.md` with detailed instructions covering:

- **Part 1 -- CPU-only VPS:** JupyterHub installation and systemd service setup, Kafka + Zookeeper as persistent system services, notebook modifications, expected training time: 40-75 minutes
- **Part 2 -- GPU VPS (RTX 4000, 16 GB VRAM):** NVIDIA driver, CUDA 12.x, and cuDNN installation, TensorFlow GPU setup with memory growth configuration, expected total time: 5-12 minutes
- Also includes troubleshooting for common issues (Kafka, GPU detection, OOM, firewall)

---

## Committing and Pushing (with SSH Troubleshooting)

**User:** commit this

**Claude:** Committed as `cde2e75`. Only staged `RUN_ON_VPS.md` -- left the untracked `dialog.md` out.

**User:** push it

**Claude:** The push failed due to an SSH key permission error.

```
git@github.com: Permission denied (publickey).
fatal: Could not read from remote repository.
```

**User:** ssh -T git@github.com

**Claude:** Your SSH key isn't set up for GitHub.

**User:** ls ~/.ssh/

**Claude:** You have an `id_ed25519` key. It's likely just not added to your SSH agent.

**User:** ssh-add ~/.ssh/id_ed25519

**Claude:** The key is passphrase-protected, which requires interactive input I can't provide. Run this in your terminal directly.

**User:** it worked, push it

**Claude:** Pushed successfully to `main`.

---

## Creating the README

**User:** we're going to work on deploying this notebook on a raw linux environment then improving and expanding it. create a README.md

**Claude:** Created `README.md` with an overview of the architecture, agents, dataset, requirements, and links to the deployment guide.

**User:** yes, commit and push it

**Claude:** Committed and pushed as `ecf1740`.

---

## Adding Mermaid Diagrams

**User:** add some mermaid diagrams to the readme in the forest theme that explain the dependencies in the notebook, then others that document how it works in terms of process and data flows

**Claude:** Added five Mermaid diagrams (all using the forest theme) to the README:

1. **Notebook Dependency Graph** -- shows which steps depend on which (e.g., Steps 3 & 4 can run in parallel after Step 2)
2. **Software Dependency Stack** -- maps external libraries and services to the components that use them
3. **Process Flow** -- sequence diagram showing the full execution timeline across threads
4. **Data Flow** -- traces how image data transforms from raw CIFAR-10 through preprocessing, training, Kafka messages, inference, and final visualization
5. **Kafka Topic Message Schemas** -- class diagram documenting the JSON structure of requests, responses, and judge results

---

## Creating the NEXT_STEPS Document

**User:** create a file called NEXT_STEPS.md where you create several sections. The first should explain how you would improve the notebook doing exactly what it is doing now. The next section should describe options for improving the notebook to take on new capabilities and demonstrate new concepts. Be sure to explain the value of these enhancements. The third section should explain how to implement the enhancements

**Claude:** Created `NEXT_STEPS.md` with three sections:

**Section 1 -- Improving the current notebook (10 items):**
- Fix the ResNet augmentation bug (the `RandomFlip` output is silently discarded)
- Add richer data augmentation
- Use the full test set for evaluation
- Add confusion matrices and training history plots
- Improve the Judge's selection strategy beyond "pick highest confidence"
- Add reproducibility seeds, model save/load, error handling, and decouple Kafka setup

**Section 2 -- New capabilities (10 options):**
- Add a third agent (MobileNet with transfer learning)
- Weighted ensemble / soft voting
- Live Streamlit dashboard
- Custom image upload
- Higher-resolution datasets (CIFAR-100, Tiny ImageNet)
- Docker Compose containerization
- Swap Kafka for Redis/RabbitMQ
- Model versioning and A/B testing
- Agent-to-agent negotiation
- Statistical analysis with McNemar's test

**Section 3 -- Implementation details** with working code for each enhancement.

---

## Editing Notebooks Directly

**User:** are you able to directly edit the notebook file?

**Claude:** Yes, I can edit the notebook directly using the `NotebookEdit` tool. It lets me:

- **Replace** the contents of any cell
- **Insert** new cells at any position
- **Delete** cells

Each cell is referenced by its ID (e.g., `cell-0`, `cell-1`, etc.). I can also change cell types between `code` and `markdown`.

The main limitation is that I replace entire cell contents at once rather than making surgical edits within a cell -- so for large cells, I'd rewrite the full cell source.

---

## Creating Stage 1 and Stage 2 Notebooks

**User:** create a new file based on the current one, prefix the name with stage1_ and make the changes you recommend in Stage 1 in the document. Create another copy prefixed with stage2_ that carries on the work described in stage 2 in the NEXT_STEPS

**Claude:** Created both notebooks:

### `stage1_image_class.ipynb` -- All 10 Stage 1 improvements:

- Fixed ResNet augmentation bug (`inputs` -> `x`)
- Added rotation, zoom, translation augmentation to both models
- Full 10,000-image test set evaluation
- Confusion matrices + classification reports for both models
- Training history plots (loss/accuracy curves)
- Improved Judge: agreement check + soft voting fallback
- Random seeds (`np.random.seed(42)`, `tf.random.set_seed(42)`)
- Model save/load to skip retraining
- Kafka decoupled (health check only, no inline install/restart)
- Error handling in all agents with try/except

### `stage2_image_class.ipynb` -- Builds on Stage 1, adds:

- MobileNetV2 as a third agent (with training, evaluation, confusion matrix)
- 3-agent soft voting ensemble in the Judge
- Updated orchestration for 3 agent threads
- 4-bar accuracy chart (CNN, ResNet, MobileNet, Ensemble)
- CSV export of all classification results
- McNemar's statistical tests (CNN vs ResNet, CNN vs MobileNet)
- Inference time box plot comparison

Both notebooks have outputs cleared and are ready to run. The README files table was updated to include them.

Committed and pushed as `d50cbe2`.
