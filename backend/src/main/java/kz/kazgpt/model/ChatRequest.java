package kz.kazgpt.model;

import java.util.List;

public record ChatRequest(
        String message,
        List<Message> history,
        String model
) {
    public String resolvedModel() {
        return (model == null || model.isBlank()) ? "base" : model;
    }

    public List<Message> safeHistory() {
        return history == null ? List.of() : history;
    }
}
