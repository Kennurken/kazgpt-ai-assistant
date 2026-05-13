package kz.kazgpt.controller;

import kz.kazgpt.config.KazGptProperties;
import kz.kazgpt.model.ChatRequest;
import kz.kazgpt.service.ChatService;
import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;

import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class ChatController {

    private final ChatService chatService;
    private final KazGptProperties props;
    private final Instant startTime = Instant.now();
    private final WebClient probe = WebClient.builder().build();

    public ChatController(ChatService chatService, KazGptProperties props) {
        this.chatService = chatService;
        this.props = props;
    }

    @PostMapping(value = "/chat/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<String> stream(@RequestBody ChatRequest request) {
        return chatService.streamChat(request);
    }

    @GetMapping("/models")
    public Map<String, Object> models() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("available", props.getModels().keySet());
        out.put("default", "base");
        out.put("details", props.getModels());
        return out;
    }

    @GetMapping("/health")
    public Mono<Map<String, Object>> health() {
        Map<String, Object> base = new LinkedHashMap<>();
        base.put("status", "ok");
        base.put("version", props.getVersion());
        base.put("uptimeSeconds", Instant.now().getEpochSecond() - startTime.getEpochSecond());
        base.put("demoMode", props.isDemoMode());

        Mono<Boolean> ollamaUp = ping(props.getModels().get("base").getUrl() + "/api/tags");
        Mono<Boolean> mlxUp = ping(props.getModels().get("v2").getUrl() + "/v1/models");

        return Mono.zip(ollamaUp, mlxUp).map(t -> {
            base.put("ollamaUp", t.getT1());
            base.put("mlxServerUp", t.getT2());
            return base;
        });
    }

    private Mono<Boolean> ping(String url) {
        return probe.get().uri(url).retrieve().toBodilessEntity()
                .map(r -> true)
                .timeout(java.time.Duration.ofSeconds(1))
                .onErrorReturn(false);
    }
}
