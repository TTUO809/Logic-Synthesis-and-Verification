#!/usr/bin/env bash
# Install ABC from source and set PATH for LSV PA3
# Run once: bash setup_abc.sh
# After running, activate env: conda activate LSV_PA3 && source ~/.bashrc

set -e

ABC_DIR="$HOME/abc"

if [ ! -d "$ABC_DIR" ]; then
    git clone https://github.com/berkeley-abc/abc.git "$ABC_DIR"
fi

cd "$ABC_DIR"
make -j$(nproc) 2>&1 | tail -5

echo "ABC built at $ABC_DIR/abc"

# Add to ~/.bashrc if not already there
if ! grep -q "abc # LSV_PA3" ~/.bashrc; then
    echo "export PATH=$ABC_DIR:\${PATH} # LSV_PA3" >> ~/.bashrc
    echo "Added ABC to PATH in ~/.bashrc"
fi

echo "Done. Run: source ~/.bashrc"
echo "Verify: abc -q 'quit'"
