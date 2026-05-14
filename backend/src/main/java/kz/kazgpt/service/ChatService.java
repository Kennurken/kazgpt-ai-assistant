package kz.kazgpt.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import kz.kazgpt.config.KazGptProperties;
import kz.kazgpt.model.CachedResponse;
import kz.kazgpt.model.ChatRequest;
import kz.kazgpt.model.Message;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

@Service
public class ChatService {

    private static final Logger log = LoggerFactory.getLogger(ChatService.class);

    private final KazGptProperties props;
    private final CacheService cacheService;
    private final ObjectMapper mapper = new ObjectMapper();
    private final WebClient.Builder webClientBuilder;

    public ChatService(KazGptProperties props, CacheService cacheService, WebClient.Builder webClientBuilder) {
        this.props = props;
        this.cacheService = cacheService;
        this.webClientBuilder = webClientBuilder;
    }

    public Flux<String> streamChat(ChatRequest req) {
        if (props.isDemoMode()) {
            Optional<CachedResponse> cached = cacheService.findMatch(req.message());
            if (cached.isPresent()) {
                log.info("Demo-mode cache hit for: {}", truncate(req.message(), 50));
                return streamFromCache(cached.get().answer());
            }
        }

        String modelKey = req.resolvedModel();
        KazGptProperties.ModelConfig cfg = props.getModels().get(modelKey);
        if (cfg == null) {
            log.warn("Unknown model {}, falling back to base", modelKey);
            cfg = props.getModels().get("base");
        }

        WebClient client = webClientBuilder.baseUrl(cfg.getUrl()).build();
        List<Message> messages = buildMessages(req);

        if ("mlx".equalsIgnoreCase(cfg.getRuntime())) {
            return streamOpenAI(client, cfg.getName(), messages);
        }
        return streamOllama(client, cfg.getName(), messages);
    }

    private Flux<String> streamFromCache(String answer) {
        String[] tokens = answer.split("(?<=\\G.{3})");
        return Flux.fromArray(tokens)
                .delayElements(Duration.ofMillis(40));
    }

    private List<Message> buildMessages(ChatRequest req) {
        List<Message> msgs = new ArrayList<>();
        msgs.add(Message.system(props.getSystemPrompt()));
        msgs.addAll(req.safeHistory());
        msgs.add(Message.user(req.message()));
        return msgs;
    }

    private Map<String, Object> generationOptions() {
        // Ollama options: см. https://github.com/ollama/ollama/blob/main/docs/api.md#parameters
        // Все параметры подбираем для борьбы с loops + сохранения краткости (KazGPT system-prompt).
        var g = props.getGeneration();
        Map<String, Object> opts = new LinkedHashMap<>();
        opts.put("temperature", g.getTemperature());
        opts.put("top_p", g.getTopP());
        opts.put("top_k", g.getTopK());
        if (g.getMinP() > 0) opts.put("min_p", g.getMinP());
        opts.put("repeat_penalty", g.getRepeatPenalty());
        opts.put("repeat_last_n", g.getRepeatLastN());
        opts.put("num_predict", g.getMaxTokens());
        opts.put("num_ctx", g.getNumCtx());
        if (g.getPresencePenalty() != 0) opts.put("presence_penalty", g.getPresencePenalty());
        if (g.getFrequencyPenalty() != 0) opts.put("frequency_penalty", g.getFrequencyPenalty());
        if (g.getStop() != null && !g.getStop().isEmpty()) opts.put("stop", g.getStop());
        return opts;
    }

    private Flux<String> streamOllama(WebClient client, String modelName, List<Message> messages) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", modelName);
        body.put("messages", messages.stream().map(m -> Map.of("role", m.role(), "content", m.content())).toList());
        body.put("stream", true);
        body.put("options", generationOptions());

        return client.post()
                .uri("/api/chat")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToFlux(String.class)
                .mapNotNull(this::extractOllamaContent)
                .filter(s -> !s.isEmpty())
                .onErrorResume(e -> {
                    log.error("Ollama stream error: {}", e.getMessage());
                    return Flux.just("\n\n[Кешіріңіз, қате орын алды. Қайталап көріңіз.]");
                });
    }

    private Flux<String> streamOpenAI(WebClient client, String modelName, List<Message> messages) {
        // OpenAI-совместимый API (MLX server, vLLM, llama.cpp с openai-mode).
        // Поддерживает только подмножество параметров — repeat_penalty/top_k тут не пробрасываем.
        var g = props.getGeneration();
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("model", modelName);
        body.put("messages", messages.stream().map(m -> Map.of("role", m.role(), "content", m.content())).toList());
        body.put("stream", true);
        body.put("temperature", g.getTemperature());
        body.put("top_p", g.getTopP());
        body.put("max_tokens", g.getMaxTokens());
        if (g.getPresencePenalty() != 0) body.put("presence_penalty", g.getPresencePenalty());
        if (g.getFrequencyPenalty() != 0) body.put("frequency_penalty", g.getFrequencyPenalty());
        if (g.getStop() != null && !g.getStop().isEmpty()) body.put("stop", g.getStop());

        return client.post()
                .uri("/v1/chat/completions")
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body)
                .retrieve()
                .bodyToFlux(String.class)
                .mapNotNull(this::extractOpenAIContent)
                .filter(s -> !s.isEmpty())
                .onErrorResume(e -> {
                    log.error("MLX/OpenAI stream error: {}", e.getMessage());
                    return Flux.just("\n\n[v2 моделі қол жетімсіз. Базалық моделді таңдаңыз.]");
                });
    }

    private String extractOllamaContent(String line) {
        try {
            if (line == null || line.isBlank()) return "";
            JsonNode node = mapper.readTree(line);
            JsonNode msg = node.get("message");
            if (msg != null && msg.get("content") != null) {
                return msg.get("content").asText();
            }
            return "";
        } catch (Exception e) {
            return "";
        }
    }

    private String extractOpenAIContent(String line) {
        try {
            if (line == null || line.isBlank()) return "";
            String json = line.startsWith("data:") ? line.substring(5).trim() : line.trim();
            if ("[DONE]".equals(json)) return "";
            JsonNode node = mapper.readTree(json);
            JsonNode choices = node.get("choices");
            if (choices != null && choices.isArray() && choices.size() > 0) {
                JsonNode delta = choices.get(0).get("delta");
                if (delta != null && delta.get("content") != null) {
                    return delta.get("content").asText();
                }
            }
            return "";
        } catch (Exception e) {
            return "";
        }
    }

    private static String truncate(String s, int n) {
        if (s == null) return "";
        return s.length() <= n ? s : s.substring(0, n) + "...";
    }
}
