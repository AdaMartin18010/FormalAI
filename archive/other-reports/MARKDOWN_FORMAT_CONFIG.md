# Markdown 保存时自动格式化配置

## 📋 配置文件清单

本项目包含以下配置文件，用于实现 Markdown 文件保存时自动格式化：

1. **`.vscode/settings.json`** - VS Code 编辑器设置
2. **`.prettierrc.json`** - Prettier 格式化规则
3. **`.editorconfig`** - 跨编辑器统一配置

---

## ⚙️ 快速配置

### 步骤 1：安装 Prettier 扩展

在 VS Code 中安装 **Prettier - Code formatter** 扩展：

- 扩展 ID：`esbenp.prettier-vscode`
- 或通过命令面板：`Ctrl+Shift+P` → 输入 "Extensions: Install Extensions" → 搜索 "Prettier"

### 步骤 2：验证配置

配置文件已创建，保存任意 `.md` 文件即可自动格式化。

---

## 🔧 配置详情

### VS Code 设置（`.vscode/settings.json`）

```json
{
  "[markdown]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  }
}
```

**主要功能**：

- ✅ 保存时自动格式化
- ✅ 删除尾随空格
- ✅ 统一行尾符（LF）
- ✅ 文件末尾自动换行

### Prettier 配置（`.prettierrc.json`）

```json
{
  "printWidth": 100,
  "tabWidth": 2,
  "useTabs": false,
  "proseWrap": "preserve"
}
```

**主要规则**：

- 行宽：100 字符
- 缩进：2 个空格
- Markdown：保持原始换行（不破坏表格和列表）

### EditorConfig（`.editorconfig`）

```text
[*]
charset = utf-8
end_of_line = lf
indent_size = 2
```

**统一标准**：

- 编码：UTF-8
- 行尾：LF（Unix 风格）
- 缩进：2 个空格

---

## 🚀 使用方法

### 自动格式化

保存文件（`Ctrl+S`）时自动格式化。

### 手动格式化

- **快捷键**：`Shift+Alt+F`（Windows/Linux）或 `Shift+Option+F`（Mac）
- **命令**：`Ctrl+Shift+P` → "Format Document"

---

## ❓ 常见问题

### Q1: 保存时未自动格式化？

**检查清单**：

1. ✅ Prettier 扩展已安装
2. ✅ `.vscode/settings.json` 存在且配置正确
3. ✅ `.prettierrc.json` 存在
4. ✅ 重启 VS Code

### Q2: Markdown 表格被破坏？

**解决方案**：

- 配置中已设置 `proseWrap: "preserve"`，应保持原始格式
- 如仍有问题，检查 Prettier 版本是否最新

### Q3: 格式化规则不符合预期？

**解决方案**：

1. 检查 `.prettierrc.json` 配置
2. 确保使用项目级配置（而非用户级）
3. 查看 VS Code 输出面板的 Prettier 日志

---

## 📝 格式化规则说明

### 自动执行的格式化操作

1. **删除尾随空格**：行末的多余空格
2. **统一行尾符**：Windows（CRLF）→ Unix（LF）
3. **文件末尾换行**：确保文件以换行符结尾
4. **代码块格式化**：格式化代码块内的代码（如果支持）

### 不会修改的内容

- ✅ Markdown 表格格式（保持原样）
- ✅ 列表缩进（保持原样）
- ✅ 标题层级（保持原样）
- ✅ 链接和图片格式（保持原样）

---

## 🔍 验证配置

创建测试文件 `test.md`：

```markdown
# 测试

文本末尾有空格
另一段文本

- 列表项
```

保存后应自动：

- ✅ 删除 "文本末尾有空格" 后的空格
- ✅ 文件末尾添加换行符

---

## 📚 相关文档

详细配置说明请参考：

- `docs/0-总览与导航/0.5-Markdown格式化配置说明.md`

---

**配置完成时间**：2025-01-XX
**最后更新**：2025-01-XX
