# cheems 设备连接信息

## 设备概况

| 项 | 值 |
|----|----|
| 主机名 | `cheems` |
| Tailscale 地址 | `cheems.tail10d158.ts.net` → `100.104.57.18` |
| Tailscale 账号 | `ouyang.xiao091130@` |
| 系统 | Windows 10 (10.0.22621.4317) |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB 显存 |
| SSH 账号 | `ai training`（用户名含空格,引用时必须加引号） |
| SSH 密码 | 已移除；请使用环境变量或本地受保护配置 |
| 主目录 | `C:\Users\AI Training` |
| 工作目录 | `D:\training`（本轮研究 monopaper 工作目录） |

## 网络

- 走 Tailscale 内网,不直连公网 IP。
- `cheems` 是 Windows 机器,有时会 offline(曾观测到 `last seen 20m ago`)。
- ping 延迟约 358–1070ms(走 relay "hkg")。

## 认证方式

本地 **没有 `sshpass`**,但本机 Python 已装 **paramiko 5.0.0**。
因此连接走 paramiko,用密码认证,不需要交互式输入密码。

服务器支持的认证:`publickey,password,keyboard-interactive`。

## 连接示例(paramiko)

```python
import paramiko, warnings
warnings.filterwarnings("ignore")

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("cheems", username="ai training", password="REDACTED_CONFIGURE_ENV", timeout=15)

# 执行命令(注意 Windows 中文系统输出编码,统一 UTF-8 处理)
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
for cmd in ["hostname", "whoami", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"]:
    stdin, stdout, stderr = c.exec_command(cmd, timeout=30)
    out = stdout.read().decode("utf-8", "replace").strip()
    print(f"$ {cmd}\n{out}\n")
c.close()
```

## 注意事项

- `ai training` 用户名含空格;在 shell / SSH 命令里必须用引号包裹,如 `"ai training@cheems"`。
- Windows SSH 默认 shell 的 `python --version` 在会话里曾无输出,需要在审计时排查 PATH / 是否装 Python。
- `ver` 等中文 Windows 命令输出含 GBK 字符,要按 UTF-8 解码,否则会 `UnicodeEncodeError`。
- 本地这台机器是 **GTX 1650**(不是 4060);4060 在 `cheems` 上。
  本地 `D:\LoopLM` 只作为暂存/镜像,真正的 baseline 复现实验在 `cheems` 上跑。
