#!/bin/bash
# Worktree management for parallel feature development
# Usage: ./scripts/worktree.sh [command] [feature-name]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_NAME="eulex"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

show_help() {
    echo -e "${GREEN}Worktree Manager${NC}"
    echo ""
    echo "Usage: ./scripts/worktree.sh [command] [feature-name]"
    echo ""
    echo "Commands:"
    echo "  new <name>      Create new worktree for feature (feature/<name> branch)"
    echo "  hotfix <name>   Create new worktree for hotfix (hotfix/<name> branch)"
    echo "  repair <name>   Fix symlinks for existing worktree"
    echo "  list            List all active worktrees"
    echo "  done <name>     Merge feature and remove worktree"
    echo "  remove <name>   Remove worktree without merging"
    echo "  sync <name>     Merge latest main into worktree branch"
    echo "  sync --all      Sync all active worktrees with main"
    echo "  open <name>     Print path to worktree (for cd)"
    echo ""
    echo "Examples:"
    echo "  ./scripts/worktree.sh new dark-mode"
    echo "  ./scripts/worktree.sh hotfix crash-on-empty-query"
    echo "  ./scripts/worktree.sh repair dark-mode  # fix missing symlinks"
    echo "  cd \$(./scripts/worktree.sh open dark-mode)"
    echo "  ./scripts/worktree.sh done dark-mode"
}

worktree_path() {
    local name=$1
    echo "$PROJECT_ROOT/../$PROJECT_NAME-$name"
}

symlink_if_missing() {
    # Create symlink only if target exists in main and link doesn't exist in worktree
    local src=$1 dst=$2 label=$3
    if [ -e "$src" ] && [ ! -e "$dst" ]; then
        ln -s "$src" "$dst"
        echo -e "  ${GREEN}✓${NC} $label"
    elif [ -L "$dst" ]; then
        echo -e "  ${YELLOW}~${NC} $label (already linked)"
    fi
}

find_docs_dir() {
    # Find docs dir for a feature regardless of status prefix
    local feature=$1
    for prefix in INPROGRESS_ DONE_ BACKLOG_ ""; do
        local candidate="$PROJECT_ROOT/docs/${prefix}${feature}"
        if [ -d "$candidate" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

rename_docs_dir() {
    # git mv docs dir to new status prefix. Idempotent + commits.
    local feature=$1 new_prefix=$2
    local current
    current=$(find_docs_dir "$feature") || return 0
    local target="$PROJECT_ROOT/docs/${new_prefix}${feature}"
    if [ "$current" = "$target" ]; then
        echo -e "  ${GREEN}~${NC} docs/${new_prefix}${feature} (already correct)"
        return 0
    fi
    echo -e "  ${YELLOW}→${NC} $(basename "$current") → ${new_prefix}${feature}"
    git -C "$PROJECT_ROOT" mv "$current" "$target"
    git -C "$PROJECT_ROOT" commit -q -m "docs(docs): mark ${feature} as ${new_prefix%_}"
}

setup_shared_resources() {
    # Symlink shared resources from main project into worktree
    # These are gitignored files/dirs that worktrees need at runtime
    local path=$1

    symlink_if_missing "$PROJECT_ROOT/.venv" "$path/.venv" ".venv"
    symlink_if_missing "$PROJECT_ROOT/.env" "$path/.env" ".env"

    # Data: symlink gitignored subdirs (not entire data/ — it has git-tracked files)
    symlink_if_missing "$PROJECT_ROOT/data/vector_store" "$path/data/vector_store" "data/vector_store/"

    symlink_if_missing "$PROJECT_ROOT/.claude/settings.local.json" "$path/.claude/settings.local.json" ".claude/settings.local.json"
    # CLAUDE.md is git-tracked — worktrees get it from checkout, no symlink needed
    symlink_if_missing "$PROJECT_ROOT/ui_react/frontend/node_modules" "$path/ui_react/frontend/node_modules" "ui_react/frontend/node_modules/"
}

cmd_new() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}"
        echo "Usage: ./scripts/worktree.sh new <feature-name>"
        exit 1
    fi

    local path=$(worktree_path "$name")
    local branch="feature/$name"

    if [ -d "$path" ]; then
        echo -e "${YELLOW}Worktree already exists at $path${NC}"
        exit 1
    fi

    # Mark docs as in-progress BEFORE creating worktree (so worktree inherits the rename)
    rename_docs_dir "$name" "INPROGRESS_"

    echo -e "${GREEN}Creating worktree for '$name'...${NC}"
    git worktree add -b "$branch" "$path" main

    echo -e "${GREEN}Setting up shared resources...${NC}"
    setup_shared_resources "$path"

    echo ""
    echo -e "${GREEN}✓ Worktree created${NC}"
    echo -e "  Path:   $path"
    echo -e "  Branch: $branch"
    echo ""
    echo -e "To start working:"
    echo -e "  ${YELLOW}cd $path${NC}"
    echo -e "  ${YELLOW}claude${NC}  # or open VS Code here"
}

cmd_hotfix() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Hotfix name required${NC}"
        echo "Usage: ./scripts/worktree.sh hotfix <name>"
        exit 1
    fi

    local path=$(worktree_path "hotfix-$name")
    local branch="hotfix/$name"

    if [ -d "$path" ]; then
        echo -e "${YELLOW}Worktree already exists at $path${NC}"
        exit 1
    fi

    echo -e "${GREEN}Creating hotfix worktree for '$name'...${NC}"
    git worktree add -b "$branch" "$path" main

    echo -e "${GREEN}Setting up shared resources...${NC}"
    setup_shared_resources "$path"

    echo ""
    echo -e "${GREEN}✓ Hotfix worktree created${NC}"
    echo -e "  Path:   $path"
    echo -e "  Branch: $branch"
    echo ""
    echo -e "To start working:"
    echo -e "  ${YELLOW}cd $path${NC}"
    echo -e "  ${YELLOW}claude${NC}  # or open VS Code here"
}

cmd_list() {
    echo -e "${GREEN}Active worktrees:${NC}"
    git worktree list
}

cmd_open() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}" >&2
        exit 1
    fi

    local path=$(worktree_path "$name")
    if [ ! -d "$path" ]; then
        echo -e "${RED}Error: Worktree '$name' does not exist${NC}" >&2
        exit 1
    fi

    echo "$path"
}

cmd_done() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}"
        exit 1
    fi

    # Auto-detect: try feature path first, then hotfix path
    local path branch
    if [ -d "$(worktree_path "$name")" ]; then
        path=$(worktree_path "$name")
        branch="feature/$name"
    elif [ -d "$(worktree_path "hotfix-$name")" ]; then
        path=$(worktree_path "hotfix-$name")
        branch="hotfix/$name"
    else
        echo -e "${RED}Error: No worktree found for '$name' (tried feature and hotfix)${NC}"
        exit 1
    fi

    echo -e "${GREEN}Merging and cleaning up '$name' ($branch)...${NC}"

    # Switch to main in the main worktree
    cd "$PROJECT_ROOT"
    git checkout main 2>/dev/null || true
    git pull origin main 2>/dev/null || true

    # Step 1: Merge (skip if already merged)
    if git branch --merged main | grep -q "$branch"; then
        echo -e "${YELLOW}Branch $branch already merged to main, skipping merge${NC}"
    else
        echo -e "${YELLOW}Merging $branch into main...${NC}"
        if ! git merge "$branch" --no-ff -m "Merge $branch"; then
            echo -e "${RED}Merge failed. Resolve conflicts, then re-run: ./scripts/worktree.sh done $name${NC}"
            exit 1
        fi
    fi

    # Step 2: Mark docs as done (idempotent — rename_docs_dir skips if already correct)
    rename_docs_dir "$name" "DONE_"

    # Step 3: Push (skip if nothing to push)
    if [ -n "$(git rev-list origin/main..main 2>/dev/null)" ]; then
        echo -e "${YELLOW}Pushing to remote...${NC}"
        if ! git push origin main; then
            echo -e "${RED}Push failed. Fix the issue, then re-run: ./scripts/worktree.sh done $name${NC}"
            echo -e "${YELLOW}Merge is complete locally — only push, worktree cleanup, and branch deletion remain.${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Main already up to date with remote, skipping push${NC}"
    fi

    # Step 4: Remove worktree and branch (only if they still exist)
    if [ -d "$path" ]; then
        echo -e "${YELLOW}Removing worktree...${NC}"
        git worktree remove "$path"
    else
        echo -e "${YELLOW}Worktree already removed${NC}"
    fi

    if git branch --list "$branch" | grep -q "$branch"; then
        git branch -d "$branch"
    fi

    # Remove checkpoint tags (local only, created by /checkpoint)
    git tag -l "checkpoint/$name/*" | xargs git tag -d 2>/dev/null || true

    echo ""
    echo -e "${GREEN}✓ '$name' merged, pushed, and cleaned up${NC}"
}

cmd_remove() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}"
        exit 1
    fi

    # Auto-detect: try feature path first, then hotfix path
    local path branch
    if [ -d "$(worktree_path "$name")" ]; then
        path=$(worktree_path "$name")
        branch="feature/$name"
    elif [ -d "$(worktree_path "hotfix-$name")" ]; then
        path=$(worktree_path "hotfix-$name")
        branch="hotfix/$name"
    else
        echo -e "${RED}Error: No worktree found for '$name' (tried feature and hotfix)${NC}"
        exit 1
    fi

    # Warn if branch has unmerged commits
    local commits=$(git log main.."$branch" --oneline 2>/dev/null | wc -l | tr -d ' ')
    if [ "$commits" -gt 0 ] 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Branch '$branch' has $commits unmerged commit(s):${NC}"
        git log main.."$branch" --oneline 2>/dev/null | head -5
        echo ""
        read -p "Remove anyway? This will lose these commits. [y/N] " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            echo -e "${GREEN}Cancelled.${NC}"
            exit 0
        fi
    fi

    echo -e "${YELLOW}Removing worktree '$name' (without merging)...${NC}"
    git worktree remove "$path" --force
    git branch -D "$branch" 2>/dev/null || true

    # Abandoned feature goes back to backlog
    rename_docs_dir "$name" "BACKLOG_"

    echo -e "${GREEN}✓ Worktree removed${NC}"
}

cmd_sync() {
    local name=$1

    # --all: sync every active worktree
    if [ "$name" = "--all" ]; then
        echo -e "${GREEN}Syncing all worktrees with main...${NC}"
        git fetch origin main 2>/dev/null || true
        local synced=0
        local wt_path="" wt_branch=""
        while IFS= read -r line; do
            if [[ "$line" == worktree\ * ]]; then
                wt_path="${line#worktree }"
            elif [[ "$line" == branch\ * ]]; then
                wt_branch="${line#branch refs/heads/}"
            elif [ -z "$line" ]; then
                # End of entry — process if we have both values
                if [ -n "$wt_path" ] && [ -n "$wt_branch" ]; then
                    if [ "$wt_path" != "$PROJECT_ROOT" ] && [ "$wt_branch" != "detached" ]; then
                        echo ""
                        echo -e "${YELLOW}Syncing $wt_branch...${NC}"
                        git -C "$wt_path" merge origin/main --no-edit && synced=$((synced + 1)) || \
                            echo -e "${RED}  ✗ Conflicts in $wt_branch — resolve manually in $wt_path${NC}"
                    fi
                fi
                wt_path="" wt_branch=""
            fi
        done < <(git worktree list --porcelain; echo)
        echo ""
        echo -e "${GREEN}✓ Synced $synced worktree(s)${NC}"
        return
    fi

    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}"
        echo "Usage: ./scripts/worktree.sh sync <name>  or  sync --all"
        exit 1
    fi

    local path=$(worktree_path "$name")

    if [ ! -d "$path" ]; then
        # Try hotfix path
        path=$(worktree_path "hotfix-$name")
        if [ ! -d "$path" ]; then
            echo -e "${RED}Error: Worktree '$name' does not exist${NC}"
            exit 1
        fi
    fi

    local branch=$(git -C "$path" branch --show-current)

    # Check for uncommitted changes
    if [ -n "$(git -C "$path" status --porcelain)" ]; then
        echo -e "${RED}Error: Worktree '$name' has uncommitted changes. Commit or stash first.${NC}"
        exit 1
    fi

    echo -e "${GREEN}Syncing '$name' ($branch) with main...${NC}"
    git fetch origin main 2>/dev/null || true
    git -C "$path" merge origin/main --no-edit

    echo ""
    echo -e "${GREEN}✓ Synced $branch with main${NC}"
}

cmd_repair() {
    local name=$1
    if [ -z "$name" ]; then
        echo -e "${RED}Error: Feature name required${NC}"
        echo "Usage: ./scripts/worktree.sh repair <feature-name>"
        exit 1
    fi

    local path=$(worktree_path "$name")

    if [ ! -d "$path" ]; then
        echo -e "${RED}Error: Worktree '$name' does not exist${NC}"
        exit 1
    fi

    echo -e "${GREEN}Repairing symlinks for '$name'...${NC}"
    setup_shared_resources "$path"
    echo ""
    echo -e "${GREEN}✓ Symlinks repaired${NC}"
}

# Main
case "${1:-help}" in
    new)     cmd_new "$2" ;;
    hotfix)  cmd_hotfix "$2" ;;
    repair)  cmd_repair "$2" ;;
    sync)    cmd_sync "$2" ;;
    list)    cmd_list ;;
    open)    cmd_open "$2" ;;
    done)    cmd_done "$2" ;;
    remove)  cmd_remove "$2" ;;
    help|*)  show_help ;;
esac
