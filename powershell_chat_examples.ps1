# EmoLLM 心理咨询 API - PowerShell 调用示例
# 服务地址: https://u895901-9072-0273df24.westc.seetacloud.com:8443

# ==================== 基础配置 ====================
$API_URL = "https://u895901-9072-0273df24.westc.seetacloud.com:8443"

# ==================== 1. 简单对话请求 ====================
function Invoke-EmoLLMChat {
    param(
        [string]$Message = "你好，我最近感到很焦虑",
        [int]$MaxTokens = 512,
        [double]$Temperature = 0.7
    )

    $headers = @{
        "Content-Type" = "application/json"
    }

    $body = @{
        messages = @(
            @{ role = "user"; content = $Message }
        )
        max_new_tokens = $MaxTokens
        temperature = $Temperature
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST -Headers $headers -Body $body
        return $response
    }
    catch {
        Write-Error "请求失败: $_"
        return $null
    }
}

# 使用示例
Write-Host "=== 简单对话示例 ===" -ForegroundColor Green
$result = Invoke-EmoLLMChat -Message "你好，我最近压力很大"
if ($result) {
    Write-Host "AI回复:" -ForegroundColor Cyan
    Write-Host $result.response
}

# ==================== 2. 多轮对话（带历史） ====================
function Invoke-EmoLLMChatWithHistory {
    param(
        [array]$Messages,  # 格式: @( @{role="user"/"assistant"; content="..."} )
        [int]$MaxTokens = 512,
        [double]$Temperature = 0.7
    )

    $headers = @{
        "Content-Type" = "application/json"
    }

    $body = @{
        messages = $Messages
        max_new_tokens = $MaxTokens
        temperature = $Temperature
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST -Headers $headers -Body $Body
        return $response
    }
    catch {
        Write-Error "请求失败: $_"
        return $null
    }
}

# 多轮对话示例
Write-Host "`n=== 多轮对话示例 ===" -ForegroundColor Green
$conversation = @(
    @{ role = "user"; content = "我最近总是失眠" },
    @{ role = "assistant"; content = "失眠确实让人很难受。能告诉我你一般几点上床，几点能睡着吗？" },
    @{ role = "user"; content = "我12点上床，但经常2-3点才能睡着" }
)

$result = Invoke-EmoLLMChatWithHistory -Messages $conversation
if ($result) {
    Write-Host "AI回复:" -ForegroundColor Cyan
    Write-Host $result.response
}

# ==================== 3. 交互式对话模式 ====================
function Start-EmoLLMInteractive {
    param(
        [int]$MaxTokens = 512
    )

    Write-Host "`n=== EmoLLM 心理咨询对话 ===" -ForegroundColor Green
    Write-Host "输入 'exit' 退出对话`n" -ForegroundColor Yellow

    $conversationHistory = @()

    while ($true) {
        $userInput = Read-Host "你"

        if ($userInput -eq "exit") {
            Write-Host "对话结束，再见！" -ForegroundColor Green
            break
        }

        # 添加用户消息到历史
        $conversationHistory += @{ role = "user"; content = $userInput }

        $headers = @{
            "Content-Type" = "application/json"
        }

        $body = @{
            messages = $conversationHistory
            max_new_tokens = $MaxTokens
            temperature = 0.7
        } | ConvertTo-Json -Depth 10

        try {
            Write-Host "AI思考中..." -ForegroundColor Gray
            $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST -Headers $headers -Body $body

            if ($response.status -eq "ok") {
                Write-Host "`nAI: " -ForegroundColor Cyan -NoNewline
                Write-Host $response.response "`n"

                # 添加AI回复到历史
                $conversationHistory += @{ role = "assistant"; content = $response.response }
            }
            else {
                Write-Host "错误: $($response.response)" -ForegroundColor Red
            }
        }
        catch {
            Write-Error "请求失败: $_"
        }
    }
}

# 启动交互式对话 (取消注释下面这行来启动)
# Start-EmoLLMInteractive

# ==================== 4. 系统提示词自定义 ====================
function Invoke-EmoLLMWithSystemPrompt {
    param(
        [string]$SystemPrompt = "你是一个专业的心理咨询师，擅长认知行为疗法",
        [string]$UserMessage = "我最近总是感到焦虑"
    )

    $headers = @{
        "Content-Type" = "application/json"
    }

    $body = @{
        messages = @(
            @{ role = "system"; content = $SystemPrompt },
            @{ role = "user"; content = $UserMessage }
        )
        max_new_tokens = 512
        temperature = 0.7
    } | ConvertTo-Json -Depth 10

    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST -Headers $headers -Body $body
        return $response
    }
    catch {
        Write-Error "请求失败: $_"
        return $null
    }
}

# 使用示例
Write-Host "`n=== 自定义系统提示词示例 ===" -ForegroundColor Green
$result = Invoke-EmoLLMWithSystemPrompt -SystemPrompt "你是一位擅长正念冥想的心理教练" -UserMessage "我想学习如何放松"
if ($result) {
    Write-Host "AI回复:" -ForegroundColor Cyan
    Write-Host $result.response
}

# ==================== 5. 带重试的请求 ====================
function Invoke-EmoLLMWithRetry {
    param(
        [string]$Message,
        [int]$MaxRetries = 3,
        [int]$DelaySeconds = 2
    )

    $headers = @{
        "Content-Type" = "application/json"
    }

    $body = @{
        messages = @(@{ role = "user"; content = $Message })
        max_new_tokens = 512
    } | ConvertTo-Json -Depth 10

    for ($i = 1; $i -le $MaxRetries; $i++) {
        try {
            Write-Host "尝试第 $i 次..." -ForegroundColor Yellow
            $response = Invoke-RestMethod -Uri "$API_URL/chat" -Method POST -Headers $headers -Body $body
            return $response
        }
        catch {
            Write-Warning "第 $i 次请求失败: $_"
            if ($i -lt $MaxRetries) {
                Start-Sleep -Seconds $DelaySeconds
            }
        }
    }

    Write-Error "请求失败，已重试 $MaxRetries 次"
    return $null
}

# ==================== 6. 流式输出（模拟） ====================
function Invoke-EmoLLMStream {
    param(
        [string]$Message
    )

    $result = Invoke-EmoLLMChat -Message $Message

    if ($result -and $result.response) {
        Write-Host "AI: " -ForegroundColor Cyan -NoNewline
        # 模拟打字机效果
        $chars = $result.response.ToCharArray()
        foreach ($char in $chars) {
            Write-Host $char -NoNewline
            Start-Sleep -Milliseconds 20
        }
        Write-Host ""
    }
}

# 使用示例 (取消注释来测试)
# Invoke-EmoLLMStream -Message "你好"

# ==================== 7. 批量请求 ====================
function Invoke-EmoLLMBatch {
    param(
        [array]$Questions  # 字符串数组
    )

    $results = @()

    foreach ($question in $Questions) {
        Write-Host "处理: $question" -ForegroundColor Yellow
        $result = Invoke-EmoLLMChat -Message $question

        if ($result) {
            $results += [PSCustomObject]@{
                Question = $question
                Answer = $result.response
                Status = $result.status
            }
        }

        # 避免请求过快
        Start-Sleep -Milliseconds 500
    }

    return $results
}

# 批量请求示例
Write-Host "`n=== 批量请求示例 ===" -ForegroundColor Green
$questions = @(
    "如何缓解工作压力？",
    "失眠怎么办？",
    "如何建立自信？"
)
$batchResults = Invoke-EmoLLMBatch -Questions $questions
$batchResults | Format-Table -AutoSize

# ==================== 8. 健康检查 ====================
function Test-EmoLLMHealth {
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/health" -Method GET
        Write-Host "服务状态:" -ForegroundColor Green
        $response | Format-List
        return $response
    }
    catch {
        Write-Error "健康检查失败: $_"
        return $null
    }
}

Write-Host "`n=== 健康检查 ===" -ForegroundColor Green
Test-EmoLLMHealth

# ==================== 使用说明 ====================
<#
.SYNOPSIS
    EmoLLM 心理咨询 API PowerShell 调用示例

.DESCRIPTION
    本脚本提供了多种方式调用 EmoLLM 心理咨询大模型 API

.EXAMPLE
    # 简单对话
    $result = Invoke-EmoLLMChat -Message "我最近很焦虑"
    Write-Host $result.response

.EXAMPLE
    # 交互式对话
    Start-EmoLLMInteractive

.EXAMPLE
    # 带历史的多轮对话
    $history = @(
        @{ role = "user"; content = "你好" },
        @{ role = "assistant"; content = "你好，有什么可以帮你的？" }
    )
    Invoke-EmoLLMChatWithHistory -Messages $history
#>
