#!/bin/bash

# Function to check if the current system is MacOS
is_macos() {
    if [[ $(uname) == "Darwin" ]]; then
        return 0  # True
    else
        return 1  # False
    fi
}

# Function to check if the current system is Ubuntu
is_ubuntu() {
    if [[ -f /etc/os-release ]] && grep -q "Ubuntu" /etc/os-release; then
        return 0  # True
    else
        return 1  # False
    fi
}

is_archlinux() {
    if [[ -f /etc/os-release ]] && grep -q "Arch Linux" /etc/os-release; then
        return 0  # True
    else
        return 1  # False
    fi
}

# Function to create or overwrite a symbolic link
create_symlink() {
    local target_file=$1
    local link_name=$2

    # Remove existing link or file if it exists
    if [ -e "$link_name" ] || [ -L "$link_name" ]; then
        rm -f "$link_name"
    fi

    # Create a new symbolic link
    ln -s "$target_file" "$link_name"
}

# Check for QT5 installation
check_qt5() {
    if pkg-config --exists Qt5Core && !is_ubuntu && !is_archlinux; then
        return 0  # QT5 is installed
    else
        return 1  # QT5 is not installed
    fi
}

# Check for CUDA and nvcc
check_cuda() {
    if command -v nvcc &> /dev/null; then
        return 0  # CUDA is installed
    else
        return 1  # CUDA is not installed
    fi
}

# Main script execution
if is_macos; then
    echo "Running on a MacOS system."

    # Define directories and target Makefile
    demo_dir="./demo"
    libpedsim_dir="./libpedsim"
    target_makefile="Makefile.macos"

    # Create symbolic links
    create_symlink "$target_makefile" "$demo_dir/Makefile"
    echo "Created/overwritten symbolic link in $demo_dir."

    create_symlink "$target_makefile" "$libpedsim_dir/Makefile"
    echo "Created/overwritten symbolic link in $libpedsim_dir."

    # Create or update config.mk
    echo "CUDA_AVAILABLE=" > "$demo_dir/config.mk"
    echo "Created config.mk with CUDA_AVAILABLE unset."

elif is_ubuntu || is_archlinux; then
    echo "Running on a Linux system."


    # Define directories
    demo_dir="./demo"
    libpedsim_dir="./libpedsim"

    # Check for QT5 and create appropriate link in demo directory
    if check_qt5; then
        demo_makefile="Makefile.qt"
    else
        demo_makefile="Makefile.noqt"
    fi

    create_symlink "$demo_makefile" "$demo_dir/Makefile"
    echo "Created/overwritten symbolic link in $demo_dir."

    # Check for CUDA and set CUDA_AVAILABLE in config.mk
    if check_cuda; then
        echo "CUDA_AVAILABLE=1" > "$demo_dir/config.mk"
        echo "CUDA detected. Updated config.mk with CUDA_AVAILABLE=1."
        create_symlink "Makefile.cuda" "$libpedsim_dir/Makefile"
        echo "Created/overwritten symbolic link to Makefile.cuda in $libpedsim_dir."
    else
        echo "CUDA_AVAILABLE=" > "$demo_dir/config.mk"
        echo "CUDA not detected. Updated config.mk with CUDA_AVAILABLE unset."
        create_symlink "Makefile.nocuda" "$libpedsim_dir/Makefile"
        echo "Created/overwritten symbolic link to Makefile.nocuda in $libpedsim_dir."
    fi

else
    echo "This system is not supported. Exiting script."
    exit 1
fi
