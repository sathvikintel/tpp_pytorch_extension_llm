# ###############################################################################
# # Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
# #                                                                             #
# # For information on the license, see the LICENSE file.                       #
# # Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# # SPDX-License-Identifier: BSD-3-Clause                                       #
# ###############################################################################
# # Author: Dhiraj Kalamkar (Intel Corp.)                                       #
# ###############################################################################

# import os
# import glob
# from setuptools import setup
# from setuptools import Command
# from setuptools import find_packages
# from torch.utils.cpp_extension import CppExtension, BuildExtension
# from subprocess import check_call, check_output
# import pathlib
# import torch
# import platform

# cwd = os.path.dirname(os.path.realpath(__file__))

# # set debug_trace_tpp = True to enable call tracing inside extension
# # export TPP_DEBUG_TRACE=0 (default) No logging (may add little overhead though)
# # export TPP_DEBUG_TRACE=1 to log TPP creation
# # export TPP_DEBUG_TRACE=2 to log previous and scope tracing
# # export TPP_DEBUG_TRACE=3 to log previous and TPP call tracing
# debug_trace_tpp = False

# libxsmm_root = os.path.join(cwd, "libxsmm")
# if "LIBXSMM_ROOT" in os.environ:
#     libxsmm_root = os.getenv("LIBXSMM_ROOT")

# xsmm_makefile = os.path.join(libxsmm_root, "Makefile")
# xsmm_include = os.path.join(libxsmm_root, "include")
# xsmm_lib = os.path.join(libxsmm_root, "lib")

# parlooper_root = os.path.join(cwd, "parlooper")
# if "PARLOOPER_ROOT" in os.environ:
#     parlooper_root = os.getenv("PARLOOPER_ROOT")

# parlooper_makefile = os.path.join(parlooper_root, "Makefile")
# parlooper_include = os.path.join(parlooper_root, "include")
# parlooper_lib = os.path.join(parlooper_root, "lib")

# if not os.path.exists(xsmm_makefile):
#     raise IOError(
#         f"{xsmm_makefile} doesn't exists! Please initialize libxsmm submodule using"
#         + "    $git submodule update --init"
#     )

# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


# class BuildMakeLib(Command):

#     description = "build C/C++ libraries using Makefile"

#     #    user_options = [
#     #        ("build-clib=", "b", "directory to build C/C++ libraries to"),
#     #        ("build-temp=", "t", "directory to put temporary build by-products"),
#     #        ("debug", "g", "compile with debugging information"),
#     #        ("force", "f", "forcibly build everything (ignore file timestamps)"),
#     #    ]
#     #
#     #    boolean_options = ["debug", "force"]

#     def initialize_options(self):
#         self.build_clib = None
#         self.build_temp = None

#         # List of libraries to build
#         self.libraries = None

#         # Compilation options for all libraries
#         self.define = None
#         self.debug = None
#         self.force = 0

#     def finalize_options(self):
#         self.set_undefined_options(
#             "build",
#             ("build_temp", "build_temp"),
#             ("debug", "debug"),
#             ("force", "force"),
#         )
#         # When building multiple third party libraries, we have to put the created lbiraries all in one place
#         # (pointed to as self.build_clib) because only this path is added to the link line for extensions
#         self.final_common_libs_dir = "third_party_libs"  # at the level of build_temp
#         self.build_clib = self.build_temp + "/" + self.final_common_libs_dir
#         self.libraries = self.distribution.libraries

#     def run(self):
#         pathlib.Path(self.build_clib).mkdir(parents=True, exist_ok=True)
#         if not self.libraries:
#             return
#         self.build_libraries(self.libraries)

#     def get_library_names(self):
#         if not self.libraries:
#             return None

#         lib_names = []
#         for (lib_name, makefile, build_args) in self.libraries:
#             lib_names.append(lib_name)
#         return lib_names

#     def get_source_files(self):
#         return []

#     def build_libraries(self, libraries):
#         for (lib_name, makefile, build_args) in libraries:
#             build_dir = pathlib.Path(self.build_temp + "/" + lib_name)
#             build_dir.mkdir(parents=True, exist_ok=True)
#             check_call(["make", "-f", makefile] + build_args, cwd=str(build_dir))
#             # NOTE: neither can use a wildcard here nor mv (since for the second library directory will already exist)
#             # This copying/hard linking assumes that the libraries are putting libraries under their respective /lib subfolder
#             check_call(
#                 ["cp", "-alf", lib_name + "/lib/.", self.final_common_libs_dir],
#                 cwd=str(self.build_temp),
#             )
#             # remove dynamic libraries to force static linking
#             check_call(
#                 ["rm", "-f", "libxsmm.so", "libparlooper.so"],
#                 cwd=str(self.build_clib),
#             )


# sources = [
#     "src/csrc/init.cpp",
#     "src/csrc/optim.cpp",
#     "src/csrc/xsmm.cpp",
#     "src/csrc/shm_coll.cpp",
#     "src/csrc/common_loops.cpp",
#     "src/csrc/qtypes.cpp",
# ]

# # AlphaFold sources
# sources += glob.glob("src/csrc/alphafold/*.cpp")

# # BERT sources
# sources += glob.glob("src/csrc/bert/pad/*.cpp")
# sources += glob.glob("src/csrc/bert/unpad/*.cpp")
# sources += glob.glob("src/csrc/bert/infer/*.cpp")

# sources += glob.glob("src/csrc/llm/*.cpp")

# # GNN sources
# sources += glob.glob("src/csrc/gnn/graphsage/*.cpp")
# sources += glob.glob("src/csrc/gnn/common/*.cpp")
# sources += glob.glob("src/csrc/gnn/gat/*.cpp")

# # DLRM sources
# sources += glob.glob("src/csrc/dlrm/*.cpp")

# extra_compile_args = ["-fopenmp", "-g", "-DLIBXSMM_DEFAULT_CONFIG"]  # , "-O0"]

# # if platform.processor() != "aarch64":
# #    extra_compile_args.append("-march=native")

# extra_compile_args.append("-march=native")

# if hasattr(torch, "float8_e5m2") and hasattr(torch, "float8_e4m3fn"):
#     extra_compile_args.append("-DPYTORCH_SUPPORTS_FLOAT8")

# if debug_trace_tpp:
#     extra_compile_args.append("-DDEBUG_TRACE_TPP")

# USE_CXX_ABI = int(torch._C._GLIBCXX_USE_CXX11_ABI)

# print("extra_compile_args = ", extra_compile_args)

# print(sources)

# setup(
#     name="tpp-pytorch-extension",
#     version="0.0.1",
#     author="Dhiraj Kalamkar",
#     author_email="dhiraj.d.kalamkar@intel.com",
#     description="Intel(R) Tensor Processing Primitives extension for PyTorch*",
#     long_description=long_description,
#     long_description_content_type="text/markdown",
#     url="https://github.com/libxsmm/tpp-pytorch-extension",
#     package_dir={"": "src"},
#     packages=find_packages("src"),
#     classifiers=[
#         "Programming Language :: Python :: 3",
#         "License :: OSI Approved :: BSD 3-Clause 'New' or 'Revised' License (BSD-3-Clause)",
#         "Operating System :: Linux",
#     ],
#     python_requires=">=3.6",
#     # install_requires=["torch>=1.4.0"],
#     scripts=["utils/run_dist.sh", "utils/run_dist_ht.sh", "utils/run_dist_numa.sh"],
#     libraries=[
#         ("xsmm", xsmm_makefile, ["CC=gcc", "CXX=g++", "AVX=2", "-j", "STATIC=1"]),
#         (
#             "parlooper",
#             parlooper_makefile,
#             [
#                 "CC=gcc",
#                 "CXX=g++",
#                 "AVX=2",
#                 f"USE_CXX_ABI={USE_CXX_ABI}",
#                 "-j",
#                 "ROOTDIR = " + parlooper_root,
#                 "LIBXSMM_ROOT=" + libxsmm_root,
#                 "PARLOOPER_COMPILER=gcc",
#             ],
#         ),
#     ],
#     ext_modules=[
#         CppExtension(
#             "tpp_pytorch_extension._C",
#             sources,
#             extra_compile_args=extra_compile_args,
#             include_dirs=[xsmm_include, parlooper_include, "{}/src/csrc".format(cwd)],
#             # library_dirs=[xsmm_lib],
#             # libraries=["xsmm"],
#         )
#     ],
#     cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
# )

###############################################################################
# Copyright (c) 2022 Intel Corporation - All rights reserved.                 #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/tpp-pytorch-extension/      #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Author: Dhiraj Kalamkar (Intel Corp.)                                       #
###############################################################################

import os
import glob
from setuptools import setup
from setuptools import Command
from setuptools import find_packages
from torch.utils.cpp_extension import CppExtension, BuildExtension
from subprocess import check_call, check_output
import pathlib
import torch
import platform

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

# --- perf-cpp setup ---
perf_cpp_include = "/home/sathvik/perf-cpp/include"
perf_cpp_lib = "/home/sathvik/perf-cpp/build/libperf-cpp.a"
# ----------------------

if not os.path.exists(xsmm_makefile):
    raise IOError(
        f"{xsmm_makefile} doesn't exists! Please initialize libxsmm submodule using"
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
sources += glob.glob("src/csrc/llm/*.cpp")
sources += glob.glob("src/csrc/gnn/graphsage/*.cpp")
sources += glob.glob("src/csrc/gnn/common/*.cpp")
sources += glob.glob("src/csrc/gnn/gat/*.cpp")
sources += glob.glob("src/csrc/dlrm/*.cpp")

extra_compile_args = [
    "-fopenmp",
    "-g",
    "-DLIBXSMM_DEFAULT_CONFIG",
    "-march=native",
    # "-D_GLIBCXX_USE_CXX11_ABI=0",  # Force same ABI as perf-cpp
]

if hasattr(torch, "float8_e5m2") and hasattr(torch, "float8_e4m3fn"):
    extra_compile_args.append("-DPYTORCH_SUPPORTS_FLOAT8")

if debug_trace_tpp:
    extra_compile_args.append("-DDEBUG_TRACE_TPP")

USE_CXX_ABI = 0  # Force same ABI as perf-cpp

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
                "USE_CXX_ABI=1",  # Force same ABI for parlooper
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
                "{}/src/csrc".format(cwd),
                perf_cpp_include,
            ],
            extra_objects=[
                perf_cpp_lib,
            ],
            libraries=["pthread"],
        )
    ],
    cmdclass={"build_ext": BuildExtension, "build_clib": BuildMakeLib},
)
