# 打包修复验证（2026-11-17，BI-V150 实例）

## 背景

按交付包 readme 的官方安装步骤 `pip install --no-build-isolation .` 安装 FlagGems 后，
`import flag_gems` 报错：

```
ModuleNotFoundError: No module named 'flag_gems.fused.DSA'
```

原因：`src/flag_gems/fused/DSA/` 没有 `__init__.py`，上游 `setup.py` 用
`find_packages("src")` 发现包，该目录不会被打进 wheel；而 `fused/__init__.py` 会
`from flag_gems.fused.DSA.bin_topk import bucket_sort_topk`。
上游 `4df52d91` 目录结构与 `setup.py` 均相同，属于上游打包缺陷，不是本项目改动引入。

## 修复

交付包在 `FlagGems/src/flag_gems/fused/DSA/__init__.py` 补一个只含注释的占位文件，
不改变运行时行为。修复后 ZIP 条目数 5,194（原 5,193），md5
`dcf95c35f40d8230fe2dc6705513ccfe`。

## 验证过程（在 BI-V150 实例上，系统 Python 3.12）

```bash
pkill -f "vllm serve"
python -m pip uninstall -y flag_gems vllm-plugin-fl
python -m pip install "setuptools>=77.0.3,<81" "setuptools-scm>=8,<10" "wheel==0.46.2"
cd XLANG2026/vllm-plugin-FL && python -m pip install --no-build-isolation .
python -m pip install "setuptools>=64,<77" "setuptools-scm>=8,<10" "wheel==0.46.2"
cd ../FlagGems && python -m pip install --no-build-isolation .
cd /workspace
python -c "import vllm_fl, flag_gems; print(vllm_fl.__file__); print(flag_gems.__file__)"
python -c "from flag_gems.fused.DSA.bin_topk import bucket_sort_topk; print('DSA import OK')"
```

输出：

```
vllm_fl: /usr/local/lib/python3.12/site-packages/vllm_fl/__init__.py
flag_gems: /usr/local/lib/python3.12/site-packages/flag_gems/__init__.py
DSA import OK
```

即按 readme 的原始安装命令（非可编辑安装）可以正常导入两个包。
