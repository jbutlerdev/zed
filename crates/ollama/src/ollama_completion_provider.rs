use crate::{ChatMessage, ChatOptions, ChatRequest, KeepAlive, OllamaModel};
use anyhow::Result;
use client::telemetry::Telemetry;
use editor::{CompletionProposal, Direction, InlayProposal, InlineCompletionProvider};
use gpui::{AppContext, EntityId, Model as GpuiModel, ModelContext, Task};
use language::{language_settings::all_language_settings, Buffer, ToOffset};
use std::{path::Path, sync::Arc, time::Duration};
use log;

pub const OLLAMA_DEBOUNCE_TIMEOUT: Duration = Duration::from_millis(75);

pub struct OllamaCompletionProvider {
    buffer_id: Option<EntityId>,
    current_completion: Option<String>,
    file_extension: Option<String>,
    pending_refresh: Task<Result<()>>,
    model: OllamaModel,
    telemetry: Option<Arc<Telemetry>>,
}

impl OllamaCompletionProvider {
    pub fn new(model: OllamaModel) -> Self {
        Self {
            buffer_id: None,
            current_completion: None,
            file_extension: None,
            pending_refresh: Task::ready(Ok(())),
            model,
            telemetry: None,
        }
    }

    pub fn with_telemetry(mut self, telemetry: Arc<Telemetry>) -> Self {
        self.telemetry = Some(telemetry);
        self
    }
}

impl InlineCompletionProvider for OllamaCompletionProvider {
    fn name() -> &'static str {
        "ollama"
    }

    fn is_enabled(
        &self,
        buffer: &GpuiModel<Buffer>,
        cursor_position: language::Anchor,
        cx: &AppContext,
    ) -> bool {
        let buffer = buffer.read(cx);
        let file = buffer.file();
        let language = buffer.language_at(cursor_position);
        let settings = all_language_settings(file, cx);
        settings.inline_completions_enabled(language.as_ref(), file.map(|f| f.path().as_ref()), cx)
    }

    fn refresh(
        &mut self,
        buffer: GpuiModel<Buffer>,
        cursor_position: language::Anchor,
        debounce: bool,
        cx: &mut ModelContext<Self>,
    ) {
        let model = self.model.clone();
        let buffer_clone = buffer.clone();
        let buffer = buffer.read(cx);
        
        // Get text before and after cursor
        let cursor_offset = cursor_position.to_offset(&buffer.snapshot());
        let buffer_text = buffer.text();
        let (prefix, suffix) = buffer_text.split_at(cursor_offset);
        let prompt = format!("{}<fim_prefix>\n{}<fim_suffix>\n{}<fim_middle>", 
            prefix.trim_end(),
            suffix.trim_start(),
            ""
        );

        log::info!("Starting refresh");
        self.pending_refresh = cx.spawn(|this, mut cx| async move {
            if debounce {
                cx.background_executor()
                    .timer(OLLAMA_DEBOUNCE_TIMEOUT)
                    .await;
            }

            let request = ChatRequest {
                model: model.name.clone(),
                messages: vec![ChatMessage::User {
                    content: prompt,
                }],
                stream: false,
                keep_alive: KeepAlive::default(),
                options: Some(ChatOptions {
                    temperature: Some(0.2),
                    ..Default::default()
                }),
                tools: vec![],
            };

            // Make the API call to Ollama
            let response = match model.chat(request).await {
                Ok(response) => response,
                Err(e) => {
                    log::error!("Error calling Ollama: {:?}", e);
                    log::info!("Model: {:?}", model);
                    return Ok(());
                }
            };

            // Extract completion from response
            let completion = match response.message {
                ChatMessage::Assistant { content, tool_calls: _} => {
                    // Remove the FIM tags if they're in the response
                    content
                        .replace("<fim_middle>", "")
                        .replace("<fim_prefix>", "")
                        .replace("<fim_suffix>", "")
                        .trim()
                        .to_string()
                }
                e => {
                    log::error!("Unexpected response from Ollama: {:?}", e);
                    return Ok(());
                }
            };

            // Only update if we got a non-empty completion
            if !completion.is_empty() {
                this.update(&mut cx, |this, cx| {
                    this.current_completion = Some(completion);
                    this.buffer_id = Some(buffer_clone.entity_id());
                    this.file_extension = buffer_clone.read(cx).file().and_then(|file| {
                        Some(
                            Path::new(file.file_name(cx))
                                .extension()?
                                .to_str()?
                                .to_string(),
                        )
                    });
                    cx.notify();
                })?;
            }

            Ok(())
        });
    }

    fn cycle(
        &mut self,
        _buffer: GpuiModel<Buffer>,
        _cursor_position: language::Anchor,
        _direction: Direction,
        _cx: &mut ModelContext<Self>,
    ) {
        // Ollama doesn't support cycling through multiple completions
    }

    fn accept(&mut self, _cx: &mut ModelContext<Self>) {
        if self.current_completion.is_some() {
            if let Some(telemetry) = self.telemetry.as_ref() {
                telemetry.report_inline_completion_event(
                    Self::name().to_string(),
                    true,
                    self.file_extension.clone(),
                );
            }
        }
        self.current_completion = None;
    }

    fn discard(&mut self, should_report_inline_completion_event: bool, _cx: &mut ModelContext<Self>) {
        if should_report_inline_completion_event && self.current_completion.is_some() {
            if let Some(telemetry) = self.telemetry.as_ref() {
                telemetry.report_inline_completion_event(
                    Self::name().to_string(),
                    false,
                    self.file_extension.clone(),
                );
            }
        }
        self.current_completion = None;
    }

    fn active_completion_text<'a>(
        &'a self,
        buffer: &GpuiModel<Buffer>,
        cursor_position: language::Anchor,
        cx: &'a AppContext,
    ) -> Option<CompletionProposal> {
        let buffer_id = buffer.entity_id();
        if Some(buffer_id) != self.buffer_id {
            return None;
        }

        self.current_completion.as_ref().map(|completion| CompletionProposal {
            inlays: vec![InlayProposal::Suggestion(
                cursor_position.bias_right(buffer.read(cx)),
                completion.clone().into(),
            )],
            text: completion.clone().into(),
            delete_range: None,
        })
    }
} 