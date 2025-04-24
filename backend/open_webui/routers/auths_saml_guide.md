# SAML认证集成指南

## 概述

本文档详细说明了Open WebUI中SAML（Security Assertion Markup Language）认证的实现机制、配置要求和工作流程。SAML是一种基于XML的开放标准，用于在身份提供商（IdP）和服务提供商（SP）之间交换身份验证和授权数据。

## SAML登录流程

Open WebUI实现了完整的SAML认证流程，包括：

1. **SP发起的登录流程**：用户从Open WebUI开始登录
2. **断言消费服务**：处理来自IdP的SAML响应
3. **元数据服务**：提供SAML配置元数据供IdP配置使用
4. **单点登出服务**：实现SP和IdP之间的同步登出

### 1. 登录流程（SP发起）

```
用户 -> Open WebUI (/saml/login) -> 重定向到IdP登录页面 -> 用户在IdP完成登录 -> IdP重定向回Open WebUI (/saml/acs)
```

登录流程详解：

1. 用户访问Open WebUI的SAML登录端点 `/saml/login`
2. 系统生成SAML请求并重定向到IdP的单点登录URL
3. 用户在IdP完成身份验证
4. IdP生成包含用户信息的SAML响应，并将用户重定向回Open WebUI的断言消费服务（ACS）端点 `/saml/acs`

### 2. 断言消费服务（ACS）

断言消费服务处理从IdP返回的SAML响应，主要步骤：

1. **验证SAML响应**：验证签名、有效期等
2. **提取用户属性**：从SAML断言中提取邮箱、用户名等信息
3. **用户管理**：
   - 如果用户不存在，创建新用户（注意：使用`Auths.insert_new_auth()`确保双表结构一致性）
   - 如果用户已存在，获取用户信息
4. **生成会话**：
   - 创建JWT令牌
   - 设置Cookie（token、saml_session_index、session_user）
5. **重定向回前端**：将用户带有token参数的URL重定向回前端应用

### 3. 元数据服务

```
IdP管理员 -> 访问 /saml/metadata -> 获取XML格式元数据 -> 在IdP中配置SP连接
```

元数据服务提供SAML配置信息，以XML格式返回所有SP配置，包括：

- 实体ID
- 断言消费服务URL
- 单点登出服务URL
- NameID格式等

### 4. 登出流程

#### SP发起登出

```
用户 -> 请求登出 (/saml/logout) -> 清除本地会话 -> 重定向到IdP登出URL -> IdP完成登出 -> IdP重定向回SP
```

1. 用户在Open WebUI请求登出
2. 系统清除本地会话（删除cookie）
3. 生成SAML登出请求并重定向到IdP
4. IdP完成登出并可能重定向回SP

#### IdP发起登出

```
用户在IdP登出 -> IdP向SP发送登出请求 (/saml/slo) -> SP清除本地会话 -> 响应IdP
```

1. 用户在IdP侧发起登出
2. IdP向Open WebUI的SLO端点发送登出请求
3. 系统处理登出请求并清除本地会话
4. 返回响应给IdP

## 配置项

### 环境变量配置

要启用SAML认证，需要在`.env`文件中配置以下环境变量：

```
# 启用SAML认证
ENABLE_SAML=true

# IdP配置
SAML_IDP_ENTITY_ID=<IdP实体ID>
SAML_IDP_SSO_URL=<IdP单点登录URL>
SAML_IDP_SLO_URL=<IdP单点登出URL>
SAML_IDP_CERT=<IdP证书>

# SP配置
SAML_SP_ENTITY_ID=<服务提供商实体ID>
SAML_SP_ACS_URL=<断言消费服务URL>
SAML_SP_SLO_URL=<单点登出服务URL>
```

### 配置项详解

| 配置项 | 说明 | 示例 |
|--------|------|------|
| ENABLE_SAML | 是否启用SAML认证 | `true` |
| SAML_IDP_ENTITY_ID | IdP实体ID，通常是IdP提供的URL | `https://sts.windows.net/8fd9dcbc-a1cd-example/` |
| SAML_IDP_SSO_URL | IdP单点登录URL，用户认证的地址 | `https://login.microsoftonline.com/8fd9dcbc-a1cd-example/saml2` |
| SAML_IDP_SLO_URL | IdP单点登出URL，用户注销的地址 | `https://login.microsoftonline.com/8fd9dcbc-a1cd-example/saml2/logout` |
| SAML_IDP_CERT | IdP证书，用于验证SAML响应签名 | `-----BEGIN CERTIFICATE-----\nMIIDBTCCAe2gAwIBA...` |
| SAML_SP_ENTITY_ID | SP实体ID，唯一标识Open WebUI服务 | `https://your-openwebui.example.com/saml/metadata` |
| SAML_SP_ACS_URL | SP断言消费服务URL | `https://your-openwebui.example.com/saml/acs` |
| SAML_SP_SLO_URL | SP单点登出服务URL | `https://your-openwebui.example.com/saml/slo` |

## SAML认证集成步骤

1. **配置环境变量**
   - 在`.env`文件中设置所有必要的SAML配置项
   - 设置`ENABLE_SAML=true`启用SAML认证

2. **获取SP元数据**
   - 启动Open WebUI服务
   - 访问`/saml/metadata`端点获取SP元数据

3. **在IdP中配置**
   - 在IdP（如Azure AD、Okta等）中创建新的应用程序
   - 上传或粘贴从`/saml/metadata`获取的元数据
   - 配置用户属性映射（确保邮箱属性正确映射）

4. **测试SAML登录**
   - 访问Open WebUI的登录页面
   - 选择SAML登录选项
   - 系统会重定向到IdP进行认证
   - 验证认证成功后是否返回Open WebUI并自动登录

## 故障排除

如果SAML认证过程中遇到问题，可以检查以下几点：

1. **检查日志**
   - Open WebUI会记录SAML认证过程的详细日志
   - 查找错误消息以定位问题

2. **验证证书**
   - 确保IdP证书格式正确且未过期
   - 检查是否正确配置了`SAML_IDP_CERT`

3. **属性映射问题**
   - 确保IdP发送了必要的用户属性（特别是邮箱）
   - 检查属性名称映射是否正确

4. **URL配置**
   - 确保所有URL配置正确，包括协议（http/https）
   - 确保没有多余的斜杠或路径

## 安全最佳实践

1. **使用HTTPS**
   - 所有SAML通信应通过HTTPS进行
   - 确保配置中的URL都使用HTTPS协议

2. **证书管理**
   - 定期更新IdP证书
   - 安全存储证书信息，避免泄露

3. **权限控制**
   - 新用户默认权限应遵循最小权限原则
   - 考虑使用IdP中的组信息自动分配角色

4. **会话安全**
   - 为JWT令牌设置合理的过期时间
   - 使用httpOnly Cookie保护令牌

## 参考资料

- [SAML 2.0标准](https://docs.oasis-open.org/security/saml/v2.0/saml-core-2.0-os.pdf)
- [Python3-SAML库文档](https://github.com/onelogin/python3-saml)
- [Azure AD SAML配置指南](https://docs.microsoft.com/en-us/azure/active-directory/manage-apps/configure-saml-single-sign-on)
- [Okta SAML配置指南](https://developer.okta.com/docs/guides/build-sso-integration/saml2/overview/)
