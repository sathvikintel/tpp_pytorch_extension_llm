###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.
# For information on the license, see the LICENSE file.
# Further information: [https://github.com/libxsmm/tpp-pytorch-extension/](https://github.com/libxsmm/tpp-pytorch-extension/)
# SPDX-License-Identifier: BSD-3-Clause
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)
###############################################################################

import os
import glob
from setuptools import setup
from setuptools import Command
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from subprocess import check_call
import pathlib
import torch

cwd = os.path.dirname(os.path.realpath(__file__))

debug_trace_tpp = False

libxsmm_root = os.path.join(cwd, "libxsmm")
if "LIBXSMM_ROOT" in os.environ:
    libxsmm_root = os.getenv("LIBXSMM_ROOT")

xsmm_makefile = os.path.join(libxsmm_root, "Makefile")
xsmm_include = os.path.join(libxsmm_root, "include")
xsmm_lib = os.path.join(libxsmm_root, "lib")

parlooper_root = os.path.join(cwd, "parlooper")
if "PARLOOPER_ROOT" in os.environ:
    parlooper_root = os.getenv("PARLOOPER_ROOT")

parlooper_makefile = os.path.join(parlooper_root, "Makefile")
parlooper_include = os.path.join(parlooper_root, "include")
parlooper_lib = os.path.join(parlooper_root, "lib")

# # --- perf-cpp setup ---
# perf_cpp_include = "/home/sathvik/perf-cpp/include"
# perf_cpp_lib = "/home/sathvik/perf-cpp/build/libperf-cpp.a"
# # ----------------------

if not os.path.exists(xsmm_makefile):
    raise IOError(
        f"{xsmm_makefile} doesn't exist! Please initialize libxsmm submodule using"
        + "    $git submodule update --init"
    )

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

class BuildMakeLib(Command):

    description = "build C/C++ libraries using Makefile"

    def initialize_options(self):
        self.build_clib = None
        self.build_temp = None
        self.libraries = None
        self.define = None
        self.debug = None
        self.force = 0

    def finalize_options(self):
        self.set_undefined_options(
            "build",
            ("build_temp", "build_temp"),
            ("debug", "debug"),
            ("force", "force"),
        )
        self.final_common_libs_dir = "third_party_libs"
        self.build_clib = self.build_temp + "/" + self.final_common_libs_dir
        self.libraries = self.distribution.libraries

    def run(self):
        pathlib.Path(self.build_clib).mkdir(parents=True, exist_ok=True)
        if not self.libraries:
            return
        self.build_libraries(self.libraries)

    def get_library_names(self):
        if not self.libraries:
            return None
        lib_names = []
        for (lib_name, makefile, build_args) in self.libraries:
            lib_names.append(lib_name)
        return lib_names

    def get_source_files(self):
        return []

    def build_libraries(self, libraries):
        for (lib_name, makefile, build_args) in libraries:
            build_dir = pathlib.Path(self.build_temp + "/" + lib_name)
            build_dir.mkdir(parents=True, exist_ok=True)
            check_call(["make", "-f", makefile] + build_args, cwd=str(build_dir))
            check_call(
                ["cp", "-alf", lib_name + "/lib/.", self.final_common_libs_dir],
                cwd=str(self.build_temp),
            )
            check_call(
                ["rm", "-f", "libxsmm.so", "libparlooper.so"],
                cwd=str(self.build_clib),
            )

USE_CXX_ABI = int(torch._C._GLIBCXX_USE_CXX11_ABI)

sources = [
    "src/csrc/init.cpp",
    "src/csrc/optim.cpp",
    "src/csrc/xsmm.cpp",
    "src/csrc/shm_coll.cpp",
    "src/csrc/common_loops.cpp",
    "src/csrc/qtypes.cpp",
]

sources += glob.glob("src/csrc/alphafold/*.cpp")
sources += glob.glob("src/csrc/bert/pad/*.cpp")
sources += glob.glob("src/csrc/bert/unpad/*.cpp")
sources += glob.glob("src/csrc/bert/infer/*.cpp")
sources += glob.glob("src/csrc/llm/*.cpp")  # include all .cpp files in llm
sources += glob.glob("src/csrc/gnn/graphsage/*.cpp")
sources += glob.glob("src/csrc/gnn/common/*.cpp")
sources += glob.glob("src/csrc/gnn/gat/*.cpp")
sources += glob.glob("src/csrc/dlrm/*.cpp")

extra_compile_args = [
    "-fopenmp",
    "-g",
    "-DLIBXSMM_DEFAULT_CONFIG",
    "-march=native",
]

if hasattr(torch, "float8_e5m2") and hasattr(torch, "float8_e4m3fn"):
    extra_compile_args.append("-DPYTORCH_SUPPORTS_FLOAT8")

if debug_trace_tpp:
    extra_compile_args.append("-DDEBUG_TRACE_TPP")

# ----> Add LDFLAGS equivalent here <----
extra_link_args = [
    "-lpthread",
    "-lnuma",
    "-flto",
    # "-L/data/sandeep/dsa_work/micro_benchmarks/library_move_pages",
    # "-lmove_page_dsa"
]

print("extra_compile_args = ", extra_compile_args)
print(sources)

setup(
    name="tpp-pytorch-extension",
    version="0.0.1",
    author="Dhiraj Kalamkar",
    author_email="dhiraj.d.kalamkar@intel.com",
    description="Intel(R) Tensor Processing Primitives extension for PyTorch*",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/libxsmm/tpp-pytorch-extension",
    package_dir={"": "src"},
    packages=find_packages("src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License (BSD-3-Clause)",
        "Operating System :: Linux",
    ],
    python_requires=">=3.6",
    scripts=["utils/run_dist.sh", "utils/run_dist_ht.sh", "utils/run_dist_numa.sh"],
    libraries=[
        ("xsmm", xsmm_makefile, ["CC=gcc", "CXX=g++", "AVX=2", "-j", "STATIC=1"]),
        (
            "parlooper",
            parlooper_makefile,
            [
                "CC=gcc",
                "CXX=g++",
                "AVX=2",
                f"USE_CXX_ABI={USE_CXX_ABI}",
                "-j",
                "ROOTDIR = " + parlooper_root,
                "LIBXSMM_ROOT=" + libxsmm_root,
                "PARLOOPER_COMPILER=gcc",
            ],
        ),
    ],
    ext_modules=[
        CppExtension(
            "tpp_pytorch_extension._C",
            sources,
            extra_compile_args=extra_compile_args,
            include_dirs=[
                xsmm_include,
                parlooper_include,
                os.path.join(cwd, "src/csrc"),
                os.path.join(cwd, "src/csrc/llm"),  # include headers in llm
                # perf_cpp_include,
            ],
            extra_objects=[
                # perf_cpp_lib,
                # "tier_infer/lib_tier_llm_dynamic_partition.so",
            ],
            # libraries=["pthread", "numa", "move_page_dsa"],
            libraries=["pthread", "numa"],
            # library_dirs=["/data/sandeep/dsa_work/micro_benchmarks/library_move_pages"],
            runtime_library_dirs=["/data/sathvik/tpp-pytorch-extension/tier_infer"],
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
)
