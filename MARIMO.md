# Marimo: A Modern Alternative to Jupyter Notebooks

## What is Marimo?

[Marimo](https://marimo.io/) is a reactive Python notebook that stores notebooks as pure Python files (`.py`) instead of JSON. When you run a cell or interact with a UI element, marimo automatically runs all dependent cells, keeping code and outputs consistent. This eliminates the hidden state problems that plague Jupyter notebooks.

Key characteristics:

- **Reactive execution** -- Change a variable in one cell and all downstream cells update automatically. No more manually re-running cells in the right order.
- **Pure Python files** -- Notebooks are saved as `.py` files, not `.ipynb` JSON. This means clean git diffs, easy code review, and no merge conflicts on cell metadata.
- **Reproducible by design** -- You cannot reference a deleted variable or execute cells out of order. A study of 10 million Jupyter notebooks found that 36% were not reproducible; marimo makes this structurally impossible.
- **Deployable as apps** -- Any marimo notebook can be served as an interactive web application with `marimo run`, no additional framework needed.
- **Executable as scripts** -- Run a notebook from the command line with `python your_notebook.py`. You can also import functions and classes from a marimo notebook into other Python programs.
- **Built-in interactivity** -- Sliders, dropdowns, tables, and plots are first-class citizens that automatically sync with the Python kernel. No callbacks or widget boilerplate.
- **AI-native editor** -- Built-in GitHub Copilot support, AI assistants, Ruff code formatting, and fast code completion.
- **Inline package management** -- Notebooks can declare their own dependencies and run in isolated virtual environments.

## Why Use Marimo Over Jupyter?

### Problems with Jupyter that Marimo Solves

| Problem | Jupyter | Marimo |
|---|---|---|
| **Hidden state** | Delete a cell, its variables persist in memory. Other cells still reference them. | Variables are removed when their defining cell is deleted. All dependent cells are invalidated. |
| **Execution order** | Cells can be run in any order, leading to notebooks that only work if you "know the right order." | Reactive execution ensures cells always run in dependency order. |
| **Reproducibility** | Run-all often fails because cells depend on state from a previous manual run. | Guaranteed reproducible. The notebook is a DAG of dependencies. |
| **Version control** | `.ipynb` files are JSON with embedded outputs, base64 images, and metadata. Git diffs are unreadable. | `.py` files produce clean, meaningful diffs. |
| **Deployment** | Requires Voila, Streamlit conversion, or other tools to serve as an app. | `marimo run notebook.py` serves it as a web app immediately. |
| **Shell commands** | `!pip install`, `!curl`, and other shell commands are common but fragile. | Shell commands are not supported. Dependencies are managed declaratively. |

### Where Jupyter Still Has an Edge

- **GitHub rendering** -- GitHub renders `.ipynb` files with outputs inline. Marimo `.py` files show up as plain Python code without outputs.
- **Stored outputs** -- Jupyter notebooks save cell outputs (plots, tables, printed text) in the file. Marimo does not store outputs, so the notebook must be re-executed to see results. This matters for slow-running notebooks or report-style documents.
- **Ecosystem maturity** -- JupyterHub, JupyterLab extensions, and nbconvert are battle-tested. Marimo is newer and its multi-user story is still evolving.
- **Colab integration** -- Google Colab runs `.ipynb` natively. Marimo has its own hosted platform ([marimo.app](https://marimo.app/)) but it is not as widely adopted.

## Converting This Notebook to Marimo

### Is It Plausible?

Yes. Marimo provides a CLI tool to convert Jupyter notebooks:

```bash
marimo convert image_class.ipynb -o image_class.py
```

However, the automated conversion will require manual cleanup because this notebook uses several patterns that are incompatible with marimo's model.

### What Needs to Change

**1. Shell commands must be removed**

Marimo does not support `!` shell commands or IPython magic. All of these must be replaced:

```python
# Jupyter (won't work in marimo):
!pip install tensorflow keras numpy matplotlib
!curl -sSOL https://archive.apache.org/dist/kafka/3.5.0/kafka_2.13-3.5.0.tgz
!./kafka_2.13-3.5.0/bin/kafka-server-start.sh -daemon ...

# Marimo alternative: use subprocess or move to external setup
import subprocess
subprocess.run(['pip', 'install', 'tensorflow', 'keras'], check=True)
```

Better yet, move all infrastructure setup (Kafka, Zookeeper) out of the notebook entirely and declare Python dependencies in the marimo notebook's inline package metadata.

**2. Global mutable state must be refactored**

Marimo enforces that each variable is defined in exactly one cell. This notebook defines and mutates several globals across cells (`cnn_model`, `resnet_model`, `x_train`, `y_train`, etc.). The training cells modify these in-place, and the agent cells reference them. This will work as long as each variable is assigned in only one cell, but the current structure may need reorganization.

**3. Threading needs careful handling**

The notebook launches daemon threads for the CNN and ResNet agents. In marimo's reactive model, re-running a cell that starts a thread could spawn duplicate agents. Thread management will need to be more explicit -- checking if an agent is already running before starting another.

**4. Outputs are not stored**

The current Jupyter notebook stores training logs, model summaries, sample images, and the final accuracy chart in the `.ipynb` file. After converting to marimo, these outputs only exist while the notebook is running. If model training takes 30+ minutes on CPU, this is a meaningful difference.

### Is It Useful?

Yes, for several reasons:

- **Reproducibility** -- The reactive execution model prevents the "run cells in the wrong order" problem that is common when returning to this notebook after time away.
- **Version control** -- The notebook becomes a clean Python file that produces readable git diffs. The current `.ipynb` file is 217 KB of JSON that is nearly impossible to review in a pull request.
- **Deployability** -- The final visualization step could be served as an interactive web app with `marimo run`, allowing others to view results without running Jupyter.
- **Iterability** -- Changing a model parameter and having all downstream cells (training, evaluation, agents, visualization) automatically update is a significant workflow improvement.
- **Import reuse** -- Helper functions like `decode_prediction()`, `get_random_test_image()`, and the model builders could be imported from the marimo notebook into other Python programs.

### Recommended Approach

Rather than converting the existing notebook in-place, the cleanest path is:

1. Run `marimo convert image_class.ipynb -o image_class.py` to get the initial structure
2. Remove all `!` shell commands and replace with proper Python or move to external setup
3. Ensure each variable is defined in exactly one cell
4. Add marimo UI elements (sliders for `num_images`, `epochs`, `batch_size`) to make the notebook interactive
5. Keep the original `.ipynb` for reference and Colab compatibility

## Running Marimo on the Ubuntu 24.04 VPS

These instructions extend the VPS setup described in [RUN_ON_VPS.md](RUN_ON_VPS.md). Kafka and system dependencies should already be installed.

### Installation

```bash
# Create a virtual environment for marimo
sudo python3 -m venv /opt/marimo
sudo /opt/marimo/bin/pip install marimo[recommended]
sudo /opt/marimo/bin/pip install tensorflow keras numpy matplotlib kafka-python pillow tqdm
```

For the GPU VPS with the RTX 4000, also install TensorFlow with CUDA support:

```bash
sudo /opt/marimo/bin/pip install tensorflow[and-cuda]
```

### Running the Editor (Development)

To use marimo as an interactive editor (like JupyterLab):

```bash
/opt/marimo/bin/marimo edit image_class.py --host 0.0.0.0 --port 8080
```

This opens the full editor with code editing, cell execution, and AI features. Access it at `http://your-vps-ip:8080`.

### Running as a Read-Only App (Production)

To serve the notebook as a web application where users can interact with UI elements but cannot edit code:

```bash
/opt/marimo/bin/marimo run image_class.py --host 0.0.0.0 --port 8080
```

### Systemd Service

Create `/etc/systemd/system/marimo.service`:

```ini
[Unit]
Description=Marimo Notebook Server
After=network.target kafka.service

[Service]
User=your_username
Environment="PATH=/opt/marimo/bin:/usr/local/bin:/usr/bin"
ExecStart=/opt/marimo/bin/marimo edit /home/your_username/notebooks/ --host 0.0.0.0 --port 8080
WorkingDirectory=/home/your_username/notebooks
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable marimo
sudo systemctl start marimo
```

### Docker Deployment

For a containerized setup, create a `Dockerfile.marimo`:

```dockerfile
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1

WORKDIR /app
COPY requirements.txt .
RUN uv pip install -r requirements.txt
RUN uv pip install marimo[recommended]

COPY image_class.py .

EXPOSE 8080

RUN useradd -m app_user
USER app_user

HEALTHCHECK --interval=30s --timeout=3s \
  CMD curl -f http://localhost:8080/health || exit 1

CMD ["marimo", "run", "image_class.py", "--host", "0.0.0.0", "-p", "8080"]
```

Build and run:

```bash
docker build -f Dockerfile.marimo -t marimo-classifier .
docker run -p 8080:8080 -it marimo-classifier
```

### Multi-User Considerations

Marimo does not have a built-in multi-user authentication system like JupyterHub. For multi-user access on the VPS:

- **Single user (simplest)** -- Run one marimo editor instance per user on different ports
- **Reverse proxy with auth** -- Put marimo behind nginx with basic auth or OAuth2 Proxy
- **JupyterHub integration** -- Use JupyterHub as the authentication and spawner layer, launching marimo instances per user
- **Docker isolation** -- Run separate Docker containers per user, each with its own marimo instance

For a reverse proxy with basic auth using nginx:

```nginx
server {
    listen 443 ssl;
    server_name notebooks.your-domain.com;

    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    auth_basic "Marimo Notebooks";
    auth_basic_user_file /etc/nginx/.htpasswd;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Firewall

Open the marimo port (if not using nginx):

```bash
sudo ufw allow 8080/tcp
```

If using nginx with HTTPS, open port 443 instead:

```bash
sudo ufw allow 443/tcp
```

## Sources

- [Marimo official site](https://marimo.io/)
- [Marimo GitHub repository](https://github.com/marimo-team/marimo)
- [Marimo vs Jupyter comparison](https://marimo.io/features/vs-jupyter-alternative)
- [Marimo deployment documentation](https://docs.marimo.io/guides/deploying/)
- [Marimo Docker deployment](https://docs.marimo.io/guides/deploying/deploying_docker/)
- [Migrating from Jupyter](https://docs.marimo.io/guides/coming_from/jupyter/)
- [Multi-user discussion](https://github.com/marimo-team/marimo/discussions/1664)
- [Why I'm switching to marimo (Towards Data Science)](https://towardsdatascience.com/why-im-making-the-switch-to-marimo-notebooks/)
