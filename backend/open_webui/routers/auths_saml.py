#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAML Authentication for Open WebUI.

这个模块提供了与SAML身份提供商（如Azure AD、Okta等）进行身份验证的功能和路由。
支持SP发起的登录流程和IdP发起的登录流程，实现单点登录(SSO)和单点登出(SLO)功能。
"""

#############################################################################
# 导入和配置部分
#############################################################################

# 标准库导入
import uuid
import time
import json
import logging
import datetime
import base64
import zlib
from typing import Dict, Any, Optional, Tuple, List

# 第三方库导入
from fastapi import APIRouter, Request, Response, Depends, HTTPException, status
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel

# SAML库导入
from onelogin.saml2.auth import OneLogin_Saml2_Auth
from onelogin.saml2.settings import OneLogin_Saml2_Settings
from onelogin.saml2.utils import OneLogin_Saml2_Utils
from onelogin.saml2.metadata import OneLogin_Saml2_Metadata

# 项目内部导入
from open_webui.constants import ERROR_MESSAGES, WEBHOOK_MESSAGES
from open_webui.env import (
    WEBUI_AUTH_COOKIE_SAME_SITE,
    WEBUI_AUTH_COOKIE_SECURE,
    SRC_LOG_LEVELS,
)
from open_webui.models.users import Users
from open_webui.models.auths import (
    Token, 
    UserResponse,
    Auths,
)
from open_webui.utils.auth import (
    create_token,
    get_current_user,
    get_password_hash,
)
from open_webui.utils.webhook import post_webhook
from open_webui.utils.misc import parse_duration
from open_webui.utils.access_control import get_permissions

# 配置日志
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])

# 创建API路由器
router = APIRouter()

#############################################################################
# 模型定义部分
#############################################################################


class SessionUserResponse(Token, UserResponse):
    """用户会话响应模型，包含令牌和权限信息"""
    expires_at: Optional[int] = None
    permissions: Optional[dict] = None


class SAMLConfigForm(BaseModel):
    """SAML配置表单模型"""
    enable_saml: Optional[bool] = None


#############################################################################
# SAML工具类
#############################################################################

class SAMLAuth:
    """
    SAML认证工具类
    
    提供与SAML相关的各种实用方法，包括请求准备、SAML初始化、编译解码、属性解析等。
    """

    @staticmethod
    async def prepare_request(request: Request) -> Dict[str, Any]:
        """
        准备SAML请求参数，从 FastAPI 请求对象提取必要的信息
        
        Args:
            request: FastAPI请求对象
            
        Returns:
            Dict[str, Any]: 用于 SAML 认证的请求参数字典
        """
        # 获取完整URL相关信息
        url_data = {
            'https': 'on' if request.url.scheme == 'https' else 'off',
            'http_host': request.headers.get('host', ''),
            'server_port': str(request.url.port or (443 if request.url.scheme == 'https' else 80)),
            'script_name': request.url.path,
            'get_data': dict(request.query_params),
            'post_data': await request.form() if request.method == 'POST' else {}
        }
        
        # 返回格式化的参数
        return {
            'http_host': url_data['http_host'],
            'server_port': url_data['server_port'],
            'script_name': url_data['script_name'],
            'get_data': url_data['get_data'],
            'post_data': url_data['post_data'],
            'https': url_data['https']
        }

    @staticmethod
    async def init_saml_auth(request: Request) -> OneLogin_Saml2_Auth:
        """
        初始化SAML认证对象
        
        Args:
            request: FastAPI请求对象
            
        Returns:
            OneLogin_Saml2_Auth: SAML认证对象
        """
        # 准备请求参数
        req = await SAMLAuth.prepare_request(request)
        saml_settings = get_saml_settings(request.app.state.config)
        
        try:
            # 创建SAML认证对象
            auth = OneLogin_Saml2_Auth(req, saml_settings)
            return auth
        except Exception as e:
            log.error(f"初始化SAML认证对象失败: {str(e)}")
            raise

    @staticmethod
    def encode_saml_request(saml_request: str) -> str:
        """
        编码SAML请求，使用DEFLATE压缩、Base64编码
        
        Args:
            saml_request: SAML请求XML字符串
            
        Returns:
            str: 编码后的SAML请求
        """
        try:
            # 将XML字符串转换为字节
            saml_request_bytes = saml_request.encode('utf-8')
            
            compressed = zlib.compress(saml_request_bytes)[2:-4]
            encoded = base64.b64encode(compressed).decode('utf-8')
            
            return encoded
        except Exception as e:
            log.error(f"编码SAML请求失败: {str(e)}")
            raise

    @staticmethod
    def decode_saml_response(saml_response: str) -> str:
        """
        解码SAML响应
        
        Args:
            saml_response: 编码的SAML响应
            
        Returns:
            str: 解码后的SAML响应XML字符串
        """
        try:
            # Base64解码
            decoded = base64.b64decode(saml_response).decode('utf-8')
            return decoded
        except Exception as e:
            log.error(f"解码SAML响应失败: {str(e)}")
            raise

    @staticmethod
    def parse_saml_attributes(auth: OneLogin_Saml2_Auth) -> Dict[str, Any]:
        """
        解析SAML响应中的属性
        
        将SAML响应中的属性转换为更易读的格式。
        使用配置中的SAML_ATTRIBUTE_MAPPING将原始属性映射到友好名称。
        
        Args:
            auth: SAML认证对象
            
        Returns:
            Dict[str, Any]: 包含用户属性的字典
        """
        # 获取原始属性
        attributes = auth.get_attributes()
        friendly_attributes = {}
        
        # 添加NameID和会话索引
        name_id = auth.get_nameid()
        if name_id:
            friendly_attributes['name_id'] = name_id
        
        session_index = auth.get_session_index()
        if session_index:
            friendly_attributes['session_index'] = session_index
        
        # 处理属性值，转换为易读的格式
        # 保留原始属性名称和值
        for key, values in attributes.items():
            if len(values) > 1:
                friendly_attributes[key] = values
            else:
                friendly_attributes[key] = values[0]
        
        # 从应用配置中获取SAML属性映射
        from open_webui.config import SAML_ATTRIBUTE_MAPPING
        attribute_mapping = SAML_ATTRIBUTE_MAPPING.value
        
        # 根据配置的属性映射添加友好别名
        for friendly_name, original_name in attribute_mapping.items():
            if original_name in attributes:
                # 处理多值属性
                if len(attributes[original_name]) > 1:
                    friendly_attributes[friendly_name] = attributes[original_name]
                else:
                    friendly_attributes[friendly_name] = attributes[original_name][0]
        
        return friendly_attributes

    @staticmethod
    def validate_saml_response(auth: OneLogin_Saml2_Auth) -> Tuple[bool, Optional[str]]:
        """
        验证SAML响应
        
        检查SAML响应是否有效，包括检查错误和认证状态。
        
        Args:
            auth: SAML认证对象
            
        Returns:
            Tuple[bool, Optional[str]]: (是否有效, 错误消息)
        """
        errors = auth.get_errors()
        if errors:
            error_reason = auth.get_last_error_reason()
            return False, error_reason
        
        if not auth.is_authenticated():
            return False, "未认证的响应"
        
        return True, None


#############################################################################
# SAML配置函数
#############################################################################

def get_saml_settings(config) -> Dict[str, Any]:
    """
    从应用配置中获取SAML设置
    
    根据应用程序配置构建SAML设置字典，包括SP（服务提供商）和IdP（身份提供商）的相关配置。
    
    Args:
        config: 应用程序配置对象
        
    Returns:
        Dict[str, Any]: SAML设置字典
    """
    settings = {
        "strict": True,
        "debug": True,
        
        # SP（服务提供商）配置
        "sp": {
            "entityId": config.SAML_SP_ENTITY_ID,
            "assertionConsumerService": {
                "url": config.SAML_SP_ACS_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
            },
            
            "singleLogoutService": {
                "url": config.SAML_SP_SLO_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            
            "NameIDFormat": "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress",
            "x509cert": "",
            "privateKey": ""
        },
        
        # IdP（身份提供商）配置
        "idp": {
            "entityId": config.SAML_IDP_ENTITY_ID,
            "singleSignOnService": {
                "url": config.SAML_IDP_SSO_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "singleLogoutService": {
                "url": config.SAML_IDP_SLO_URL,
                "binding": "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
            },
            "x509cert": config.SAML_IDP_CERT if hasattr(config, 'SAML_IDP_CERT') else ""
        },
        
        # 安全设置
        "security": {
            "nameIdEncrypted": False,
            "authnRequestsSigned": False,
            "logoutRequestSigned": False,
            "logoutResponseSigned": False,
            "signMetadata": False,
            "wantMessagesSigned": False,
            "wantAssertionsSigned": False,
            "wantNameId" : True,
            "wantNameIdEncrypted": False,
            "wantAssertionsEncrypted": False,
            "allowSingleLabelDomains": False,
            "signatureAlgorithm": "http://www.w3.org/2001/04/xmldsig-more#rsa-sha256",
            "digestAlgorithm": "http://www.w3.org/2001/04/xmlenc#sha256"
        }
    }
    
    return settings


#############################################################################
# 路由处理函数 - 登录和认证
#############################################################################

@router.get("/login")
async def login(request: Request):
    """
    初始化SAML登录流程
    
    SP发起的登录流程，将用户重定向到IdP的登录页面进行认证。
    认证成功后，IdP会将用户重定向回/saml/acs端点并提供用户信息。
    """
    # 检查SAML认证是否启用
    if not request.app.state.config.ENABLE_SAML:
        raise HTTPException(400, detail="SAML登录未启用")
    
    try:
        auth = await SAMLAuth.init_saml_auth(request)
        login_url = auth.login(
            force_authn=True, 
            # is_passive=True,
        )
        log.info(f"SAML登录请求: {login_url}")
        
        return RedirectResponse(login_url)
    except Exception as e:
        log.error(f"SAML登录初始化失败: {str(e)}")
        raise HTTPException(500, detail=f"SAML登录初始化失败: {str(e)}")


@router.post("/acs")
async def acs(request: Request, response: Response):
    """
    断言消费服务 (Assertion Consumer Service)
    
    处理从IdP返回的SAML响应，验证用户身份，创建或更新用户，并生成JWT令牌。
    这是SAML认证流程的核心部分，由IdP完成认证后调用。
    """
    if not request.app.state.config.ENABLE_SAML:
        raise HTTPException(400, detail="SAML登录未启用")
    
    try:
        #----------------------------------------------------------------------
        # 第一步：处理和验证SAML响应
        #----------------------------------------------------------------------
        auth = await SAMLAuth.init_saml_auth(request)
        
        auth.process_response()
        
        is_valid, error_msg = SAMLAuth.validate_saml_response(auth)
        if not is_valid:
            log.error(f"SAML响应验证失败: {error_msg}")
            raise HTTPException(400, detail=f"SAML响应验证失败: {error_msg}")
        
        #----------------------------------------------------------------------
        # 第二步：提取用户属性
        #----------------------------------------------------------------------
        user_data = SAMLAuth.parse_saml_attributes(auth)
        log.info(f"SAML返回的用户信息: {user_data}")
        
        # 检查部门字段是否满足要求（只允许NNIT-Department部门的用户登录）
        # 优先使用便捷别名，如果不存在则使用原始属性路径
        department = user_data.get('department', user_data.get('http://schemas.xmlsoap.org/ws/2005/05/identity/claims/department', ''))
        # if department != 'NNIT-Department':
        #     log.error(f"用户部门不匹配：{department}，仅允许NNIT-Department部门的用户登录")
        #     raise HTTPException(403, detail=f"访问被拒绝：仅允许NNIT-Department部门的用户登录")
        
        email = user_data.get('email', user_data.get('name_id', ''))
        if not email:
            log.error("SAML响应中未包含邮箱地址")
            raise HTTPException(400, detail="SAML响应中未包含邮箱地址")
        
        name = user_data.get('display_name', user_data.get('username', email.split('@')[0]))
        
        #----------------------------------------------------------------------
        # 第三步：查找或创建用户
        #----------------------------------------------------------------------
        user = Users.get_user_by_email(email.lower())
        
        # 如果用户不存在，创建新用户
        if not user:
            # 检查用户配额限制
            user_count = Users.get_num_users()
            if request.app.state.USER_COUNT and user_count >= request.app.state.USER_COUNT:
                raise HTTPException(
                    status.HTTP_403_FORBIDDEN,
                    detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
                )
            
            # 确定用户角色（第一个用户为管理员，其他用户使用配置的默认角色）
            role = "admin" if user_count == 0 else request.app.state.config.DEFAULT_USER_ROLE
            
            # 生成随机密码（SAML用户不会直接用密码登录，但需要保存哈希密码）
            password = str(uuid.uuid4())
            hashed_password = get_password_hash(password)
            
            # 这样可以同时创建Auth和User记录，确保双表结构一致性
            user = Auths.insert_new_auth(
                email=email.lower(),
                password=hashed_password,
                name=name,
                role=role,
            )
            
            # 检查用户创建是否成功
            if not user:
                raise HTTPException(500, detail=ERROR_MESSAGES.CREATE_USER_ERROR)
            
            log.info(f"通过SAML创建新用户: {email}")
        
        #----------------------------------------------------------------------
        # 第四步：生成JWT令牌和会话
        #----------------------------------------------------------------------
        expires_delta = parse_duration(request.app.state.config.JWT_EXPIRES_IN)
        expires_at = None
        if expires_delta:
            expires_at = int(time.time()) + int(expires_delta.total_seconds())
        
        token = create_token(
            data={"id": user.id},  # 仅包含用户ID作为令牌身份标识
            expires_delta=expires_delta,
        )
        
        datetime_expires_at = (
            datetime.datetime.fromtimestamp(expires_at, datetime.timezone.utc)
            if expires_at
            else None
        )
        
        #----------------------------------------------------------------------
        # 第五步：设置Cookie和会话数据
        #----------------------------------------------------------------------
        response.set_cookie(
            key="token",
            value=token,
            expires=datetime_expires_at,
            httponly=True,  # 设置为httponly增强安全性
            samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
            secure=WEBUI_AUTH_COOKIE_SECURE,
        )
        
        if user_data.get('session_index'):
            response.set_cookie(
                key="saml_session_index",
                value=user_data.get('session_index'),
                expires=datetime_expires_at,
                httponly=True,
                samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
                secure=WEBUI_AUTH_COOKIE_SECURE,
            )
        
        log.info(f"用户 {email} 通过SAML登录成功")
        
        if request.app.state.config.WEBHOOK_URL:
            post_webhook(
                request.app.state.WEBUI_NAME,
                request.app.state.config.WEBHOOK_URL,
                WEBHOOK_MESSAGES.USER_LOGIN(user.name),
                {
                    "action": "login",
                    "message": WEBHOOK_MESSAGES.USER_LOGIN(user.name),
                    "user": user.model_dump_json(exclude_none=True),
                    "login_method": "saml",
                },
            )
        
        permissions = get_permissions(
            user.id, request.app.state.config.USER_PERMISSIONS
        )
        
        log.info(f"SAML登录成功，返回用户信息: id={user.id}, email={user.email}, name={user.name}, role={user.role}")
        
        #----------------------------------------------------------------------
        # 第六步：创建用户会话并设置前端可访问Cookie
        #----------------------------------------------------------------------
        user_info = {
            "id": user.id,
            "email": user.email,
            "name": user.name,
            "role": user.role,
            "profile_image_url": user.profile_image_url or ""
        }
        
        response.set_cookie(
            key="session_user",
            value=json.dumps(user_info),
            expires=datetime_expires_at,
            httponly=False,  # 允许JavaScript访问以便前端获取用户信息
            samesite=WEBUI_AUTH_COOKIE_SAME_SITE,
            secure=WEBUI_AUTH_COOKIE_SECURE,
        )
        
        user_response = SessionUserResponse(
            id=user.id,
            email=user.email,
            name=user.name,
            role=user.role,
            profile_image_url=user.profile_image_url,
            token=token,
            token_type="bearer",
            expires_at=expires_at,
            permissions=permissions
        )
        
        #----------------------------------------------------------------------
        # 第七步：重定向到前端应用
        #----------------------------------------------------------------------
        # 构造重定向URL，包含token参数
        redirect_url = f"{request.base_url.scheme}://{request.base_url.netloc}/auth#token={token}"
        
        # 返回重定向响应，使用303状态码确保使用GET方法
        return RedirectResponse(url=redirect_url, status_code=303)
        
    except Exception as e:
        log.error(f"SAML认证处理失败: {str(e)}")
        raise HTTPException(500, detail=f"SAML认证处理失败: {str(e)}")


#############################################################################
# 路由处理函数 - 元数据和配置
#############################################################################

@router.get("/metadata")
@router.get("/metadata/")
async def metadata(request: Request):
    """
    提供SAML服务提供商(SP)元数据
    
    这个端点生成SAML元数据文档，包含此SP的所有配置和能力信息。
    身份提供商(IdP)可以使用这个元数据来配置与此SP的集成。
    通常在初始配置时，将此URL提供给IdP管理员以设置SSO集成。
    """
    log.info(f"SAML元数据请求: {request.url}")
    
    if not request.app.state.config.ENABLE_SAML:
        raise HTTPException(400, detail="SAML登录未启用")
    
    try:
        saml_settings = get_saml_settings(request.app.state.config)
        
        log.info(f"SAML设置: {saml_settings}")
        
        # 使用OneLogin库生成SAML元数据 XML
        # 注意：只传递SP(服务提供商)配置给builder方法
        metadata = OneLogin_Saml2_Metadata.builder(
            saml_settings['sp']
        )
        
        return Response(
            content=metadata,
            media_type="application/xml"  # 指定正确的媒体类型以确保IdP可以正确解析
        )
    except Exception as e:
        log.error(f"生成SAML元数据失败: {str(e)}")
        raise HTTPException(500, detail=f"生成SAML元数据失败: {str(e)}")


@router.get("/logout")
async def logout(request: Request, response: Response, user=Depends(get_current_user)):
    """
    通过SAML进行单点登出(SLO)
    
    此端点处理用户登出请求，包含两个主要步骤：
    1. 清除本地会话和身份验证Cookie
    2. 生成并发送SAML单点登出请求到IdP，使用户在IdP端也能注销
    """
    if not request.app.state.config.ENABLE_SAML:
        raise HTTPException(400, detail="SAML登录未启用")
    
    try:
        session_index = request.cookies.get("saml_session_index")
        log.info(f"用户登出请求，会话索引: {session_index}")
        
        auth = await SAMLAuth.init_saml_auth(request)
        
        #----------------------------------------------------------------------
        # 第一步：清除本地会话和Cookie
        #----------------------------------------------------------------------
        response.delete_cookie("token")
        response.delete_cookie("saml_session_index")
        response.delete_cookie("session_user")
        
        #----------------------------------------------------------------------
        # 第二步：构建SAML登出请求并重定向
        #----------------------------------------------------------------------
        logout_url = auth.logout(
            name_id=user.email,
            session_index=session_index
        )
        
        log.info(f"用户 {user.email} 登出成功，登出URL: {logout_url if logout_url else '无IdP登出URL'}")
        
        if logout_url:
            return RedirectResponse(url=logout_url)
        else:
            return RedirectResponse(url="/")
    except Exception as e:
        log.error(f"SAML登出失败: {str(e)}")
        
        response.delete_cookie("token")
        response.delete_cookie("saml_session_index")
        response.delete_cookie("session_user")
        
        return RedirectResponse(url="/")


@router.get("/slo")
async def slo(request: Request, response: Response):
    """
    单点登出服务 (Single Logout Service)
    
    此端点处理来自IdP的单点登出请求。当用户在IdP上登出时，
    IdP会发送消息到这个端点，触发应用程序端的会话清除操作。
    这是实现完整单点登出功能的必要组成部分。
    """
    if not request.app.state.config.ENABLE_SAML:
        raise HTTPException(400, detail="SAML登录未启用")
    
    try:
        log.info(f"SAML SLO请求接收: {request.url}")
        
        auth = await SAMLAuth.init_saml_auth(request)
        
        #----------------------------------------------------------------------
        # 第一步：处理登出请求
        #----------------------------------------------------------------------
        url = auth.process_slo()
        log.info(f"SAML SLO处理结果，重定向URL: {url if url else '无重定向URL'}")
        
        #----------------------------------------------------------------------
        # 第二步：清除本地会话和身份验证Cookie
        #----------------------------------------------------------------------
        response.delete_cookie("token")
        response.delete_cookie("saml_session_index")
        response.delete_cookie("session_user")
        
        #----------------------------------------------------------------------
        # 第三步：重定向到适当的页面
        #----------------------------------------------------------------------
        if url:
            log.info(f"SAML SLO成功，重定向到: {url}")
            return RedirectResponse(url=url)
        else:
            log.info("SAML SLO成功，重定向到首页")
            return RedirectResponse(url="/")
    except Exception as e:
        log.error(f"SAML SLO处理失败: {str(e)}")
        
        response.delete_cookie("token")
        response.delete_cookie("saml_session_index")
        response.delete_cookie("session_user")
        
        return RedirectResponse(url="/")


@router.get("/config")
async def get_saml_config(request: Request, user=Depends(get_current_user)):
    """
    获取SAML配置
    """
    if user.role != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    
    return {
        "ENABLE_SAML": request.app.state.config.ENABLE_SAML,
        "SAML_IDP_ENTITY_ID": request.app.state.config.SAML_IDP_ENTITY_ID,
        "SAML_IDP_SSO_URL": request.app.state.config.SAML_IDP_SSO_URL,
        "SAML_IDP_SLO_URL": request.app.state.config.SAML_IDP_SLO_URL,
        "SAML_SP_ENTITY_ID": request.app.state.config.SAML_SP_ENTITY_ID,
        "SAML_SP_ACS_URL": request.app.state.config.SAML_SP_ACS_URL,
        "SAML_SP_SLO_URL": request.app.state.config.SAML_SP_SLO_URL,
    }


@router.post("/config")
async def update_saml_config(
    request: Request, form_data: SAMLConfigForm, user=Depends(get_current_user)
):
    """
    更新SAML配置
    """
    if user.role != "admin":
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            detail=ERROR_MESSAGES.ACCESS_PROHIBITED,
        )
    
    if form_data.enable_saml is not None:
        request.app.state.config.ENABLE_SAML.value = form_data.enable_saml
    
    return {
        "ENABLE_SAML": request.app.state.config.ENABLE_SAML.value,
    }
