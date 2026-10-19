# Installing the runmedic skill

> **For humans:** this file is meant for an AI agent. Send your agent the one-line phrase from the [README](README.md#agent-skill). It installs the `runmedic` CLI with pip and copies one skill folder (markdown files only) into the agent's skills directory.
>
> **Для людей:** этот файл написан для AI-агента. Отправь агенту фразу из [README](README.ru.md#скилл-для-агентов). Он поставит CLI `runmedic` через pip и скопирует одну папку скилла (только markdown-файлы).

---

## Instructions for the agent

The user asked you to install **runmedic** from https://github.com/vincere-mori/runmedic: a CLI that diagnoses ML training runs from their logs, and a skill that teaches you to use it. Tell the user what you are going to do before running commands, and talk to them in their language.

### 1. The CLI

Install it into the Python environment the user trains in (ask if there are several, for example a project venv or conda env):

```bash
pip install git+https://github.com/vincere-mori/runmedic
```

If they use W&B, install `"runmedic[wandb] @ git+https://github.com/vincere-mori/runmedic"` instead. If they prefer not to install anything, skip this step: the skill can run the CLI through `uvx --from git+https://github.com/vincere-mori/runmedic runmedic`.

Check: `runmedic --version`.

### 2. The skill

**Claude Code:** run in the shell

```bash
claude plugin marketplace add vincere-mori/runmedic
claude plugin install runmedic@runmedic
```

If the `claude` command is not available in the shell, use the fallback below with the target directory `~/.claude/skills/runmedic`.

**Codex:** use the fallback below with the target directory `~/.agents/skills/runmedic`.

**Any other agent that supports `SKILL.md` skills:** use the fallback with your user-level skills directory. If you do not know where it is, ask the user instead of guessing.

**Fallback, copy the folder:**

1. `git clone --depth 1 https://github.com/vincere-mori/runmedic.git <tmp>/runmedic`
   (without git, download `plugins/runmedic/skills/runmedic/SKILL.md` and `plugins/runmedic/skills/runmedic/references/playbook.md` from `https://raw.githubusercontent.com/vincere-mori/runmedic/main/` keeping the folder structure)
2. Copy `plugins/runmedic/skills/runmedic` to the target directory so that it contains `SKILL.md` and `references/playbook.md`. Create parent directories if needed; replace an older version if it exists. On Windows `~` is `%USERPROFILE%`.
3. Delete the temporary clone.

### 3. Tell the user

- what was installed and where;
- they may need to restart the agent session for the skill to show up;
- how to use it: just ask, for example "why did my training blow up?", "is my model overfitting?", "why is the GPU idle?", pointing at the logs or the project. In Claude Code it can also be called as `/runmedic:runmedic`.

Do not start diagnosing anything unless the user asks.
