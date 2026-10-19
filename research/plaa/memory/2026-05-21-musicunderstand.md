# MusicUnderstand — 项目总结

> 日期: 2026-05-21
> 项目路径: H:\musicunderstand

---

## 一、已完成功能

### ✅ 情绪分析（核心模型）
- **架构**: Mel频谱 → CNN (64→128→256) → 双向LSTM (256, 2层) → 逐帧 valence/arousal
- **数据集**: DEAM (1800+ 首歌曲，每秒情绪标注)
- **训练**: 已完成，best_model.pth (12MB, CPU可运行)
- **输出**: 每0.5秒的 (Valence, Arousal) 情绪曲线
- **已验证**: 对传统音乐（昆曲）、电子音乐（Waking Shadows, RoquiRa）均能输出有意义的情绪曲线

### ✅ 乐器检测
- **路径**: `audio_utils/instrument_detector.py`
- **可识别**: piano, synth, violin, orchestra, kick_drum, snare, bass, dist_guitar, cymbal, vocal
- **方法**: 基于频段能量分析的规则系统，无需额外模型
- **输出**: 每15秒段落的乐器活动热力图

### ✅ 风格分类（知识库驱动）
- **路径**: `audio_utils/genre_analyzer.py` + `teach.py`
- **方法**: 双模式
  - **规则模式**: 基于声学特征的启发式判断
  - **学习模式**: 基于用户标注的段落进行相似度匹配
- **知识库**: 24段样本 / 12种风格（持续增长中）

### ✅ 综合分析流水线
- **路径**: `full_pipeline.py`
- **一次输出**: 情绪 + 乐器 + 风格 + 关键段落标注

### ✅ 教学系统
- **路径**: `teach.py`
- **功能**: 用户标记段落 → 保存特征指纹 → 用于新歌预测
- **存储**: `genre_knowledge.json`

---

## 二、已收录风格（12种）

| 风格 | 样本数 | 来源 |
|:-----|:-----:|:----|
| symphonic_metal | 5 | Waking Shadows + RoquiRa |
| hard_renaissance | 3 | Waking Shadows + RoquiRa |
| glitch_hop | 3 | RoquiRa |
| artcore | 2 | Waking Shadows |
| j_core | 2 | Waking Shadows |
| j_tribecore | 2 | RoquiRa |
| neurofunk | 2 | RoquiRa |
| mainstream_hardcore | 1 | RoquiRa |
| breakcore | 1 | RoquiRa |
| basscore | 1 | RoquiRa |
| speedcore | 1 | RoquiRa |
| gothic_hardcore | 1 | Waking Shadows |

---

## 三、已完成的研究（可用于扩展知识库）

### 电子音乐谱系
```
House家族 (120-130 BPM)
├── Deep House (118-125)
├── Progressive House (122-130)
├── Tech House (122-130)
├── Bass House (124-132)
├── Big Room (126-132)
└── Electro House

Techno家族 (130-150 BPM)
├── Detroit Techno (120-135)
├── Minimal Techno (124-130)
├── Melodic Techno (120-128)
├── Industrial Techno (130-145)
└── Hard Techno (140-160)

Trance家族 (128-148 BPM)
├── Progressive Trance (128-138)
├── Vocal Trance (132-138)
├── Uplifting Trance (138-145)
├── Tech Trance (138-145)
└── Psytrance (138-148)

Breakbeat家族
├── Breakbeat (110-150)
├── Dubstep (138-142, half-time)
├── Drum & Bass (170-180)
│   ├── Liquid DnB (170-176)
│   └── Neurofunk (174-180)
├── Breakcore (160-300+)
└── Trap EDM (140-150, half-time)

Hardcore家族 (150-300+ BPM)
├── Hardstyle (148-155)
├── Rawstyle (150-160)
├── Hardcore/Gabber (160-200)
│   └── J-Core ─┬─ Artcore (160-185)
│                ├─ Hard Renaissance (175-220)
│                └─ Gothic Hardcore (165-210)
├── Frenchcore (200-220)
└── Speedcore (250-1000+)
```

### 摇滚谱系
```
Classic Rock (100-140) → Blues Rock → Hard Rock
Punk (150-200) → Pop Punk / Post-Punk
Grunge (100-130)
Alternative / Indie Rock (100-140)
Progressive Rock (80-160)
Psychedelic Rock (90-130)
Shoegaze (90-120) / Post-Rock (80-140)
Emo (120-160) / Stoner Rock (60-110)
```

### 金属谱系
```
Heavy Metal (100-150)
├── Thrash (150-220)
├── Death Metal (140-240)
│   ├── Melodic Death / Technical Death / Brutal Death
├── Black Metal (130-200+)
├── Doom Metal (50-80)
├── Power Metal (140-180)
├── Progressive Metal (80-180)
├── Symphonic Metal (120-160)
├── Metalcore (120-160)
├── Deathcore (100-180)
└── Nu Metal (90-130)
```

---

## 四、已验证的测试曲目

| 曲目 | 时长 | 特点 |
|:-----|:----:|:----|
| 张军 - 霓裳羽衣 | 3:30 | 传统昆曲，情绪平缓，Aro≈0 |
| Waking Shadows (feat. Eili) | 5:45 | Artcore+Hard Renaissance+J-Core，Aro 0.35 |
| RoquiRa: Between Life and Death | 10:46 | 6个Drop/12种风格，情绪峰值在第3Drop |

---

## 五、已知局限

1. **风格分类精度不足** — 知识库只有24段样本，细分子风格区分度有限
2. **无GPU运行** — 推理速度偏慢（10分钟曲目约需20-30秒）
3. **mp3解码依赖** — 当前环境需通过librosa fallback解码，不稳定
4. **乐器检测为规则系统** — 对复杂混音的准确率有限
5. **风格标注依赖人工** — 模型不能从零识别细分风格，需要用户教

---

## 六、下一步方向

1. **扩充知识库** — 通过霖泽xxx（B站UP主）的分析视频批量获取标签
2. **分层分类器** — 先判断大族（电子/摇滚/金属）→ 再判断子风格
3. **图谱可视化** — 把全谱系图生成可交互的HTML
4. **调式/和弦检测** — 增加key/scale检测，帮助区分明暗风格
5. **LLM Interpreter** — 把情绪曲线+乐器+风格翻译成自然语言描述
