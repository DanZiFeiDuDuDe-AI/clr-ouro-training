# 研究归档约定

- 本仓库包含多个历史阶段。阅读文件、整理归档和恢复产物不授权启动历史训练脚本。
- O3 工作只限已授权的 REBASE2 failure localization；禁止 allocator、router/memory reconnect、integrated O3 evaluation 和 TestA/TestB 调参。
- Candidate B 保持冻结；Rule A threshold 必须是 `0.8504285216331482`。
- 10K standalone validation 已有通过记录，但不自动授权下游操作。历史报告中的缺口不能靠推测补齐。
- 每次远程训练完成后，必须重新连接并拉取 checkpoint、配置和指标，验证 SHA-256，保留 transfer manifest；仅远端存在的权重不能视为交付完成。
- 不覆盖不同内容的权重；新版本使用独立目录。凭据只使用运行者的环境变量或本地受保护配置，禁止提交密码、令牌、私钥和 DPAPI 文件。
- 归档文件以 `manifests/files.jsonl` 的发布哈希核对；原始哈希不同的脱敏文件见 `manifests/redactions.json`。
