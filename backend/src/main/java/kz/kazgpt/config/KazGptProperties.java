package kz.kazgpt.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

import java.util.Collections;
import java.util.List;
import java.util.Map;

@Component
@ConfigurationProperties(prefix = "kazgpt")
public class KazGptProperties {

    private String version = "1.0";
    private boolean demoMode = false;
    private String cacheFile = "classpath:cached_responses.json";
    private String systemPrompt = "";
    private Map<String, ModelConfig> models;
    private Generation generation = new Generation();

    public static class ModelConfig {
        private String url;
        private String name;
        private String runtime;

        public String getUrl() { return url; }
        public void setUrl(String url) { this.url = url; }
        public String getName() { return name; }
        public void setName(String name) { this.name = name; }
        public String getRuntime() { return runtime; }
        public void setRuntime(String runtime) { this.runtime = runtime; }
    }

    public static class Generation {
        // Базовые
        private double temperature = 0.3;
        private double topP = 0.8;
        private double repeatPenalty = 1.2;
        private int maxTokens = 512;

        // Phase 0: расширение для борьбы с loops и hallucinations.
        // Ollama поддерживает все эти параметры в options{}.
        // OpenAI (MLX server) — частично; неподдерживаемые игнорируются.
        private int topK = 40;                   // ограничивает выбор top-k токенов
        private double minP = 0.0;               // отрезает токены с вероятностью ниже min_p * max
        private int repeatLastN = 256;           // окно для repeat_penalty (дефолт Ollama=64 — мало!)
        private double presencePenalty = 0.0;    // штраф за уже встретившиеся токены (OpenAI)
        private double frequencyPenalty = 0.0;   // штраф за частоту (OpenAI)
        private int numCtx = 4096;               // размер контекста (Ollama). 4096 хватит для chat history
        private List<String> stop = Collections.emptyList();  // массив stop-токенов

        public double getTemperature() { return temperature; }
        public void setTemperature(double temperature) { this.temperature = temperature; }
        public double getTopP() { return topP; }
        public void setTopP(double topP) { this.topP = topP; }
        public double getRepeatPenalty() { return repeatPenalty; }
        public void setRepeatPenalty(double repeatPenalty) { this.repeatPenalty = repeatPenalty; }
        public int getMaxTokens() { return maxTokens; }
        public void setMaxTokens(int maxTokens) { this.maxTokens = maxTokens; }
        public int getTopK() { return topK; }
        public void setTopK(int topK) { this.topK = topK; }
        public double getMinP() { return minP; }
        public void setMinP(double minP) { this.minP = minP; }
        public int getRepeatLastN() { return repeatLastN; }
        public void setRepeatLastN(int repeatLastN) { this.repeatLastN = repeatLastN; }
        public double getPresencePenalty() { return presencePenalty; }
        public void setPresencePenalty(double presencePenalty) { this.presencePenalty = presencePenalty; }
        public double getFrequencyPenalty() { return frequencyPenalty; }
        public void setFrequencyPenalty(double frequencyPenalty) { this.frequencyPenalty = frequencyPenalty; }
        public int getNumCtx() { return numCtx; }
        public void setNumCtx(int numCtx) { this.numCtx = numCtx; }
        public List<String> getStop() { return stop; }
        public void setStop(List<String> stop) { this.stop = stop; }
    }

    public String getVersion() { return version; }
    public void setVersion(String version) { this.version = version; }
    public boolean isDemoMode() { return demoMode; }
    public void setDemoMode(boolean demoMode) { this.demoMode = demoMode; }
    public String getCacheFile() { return cacheFile; }
    public void setCacheFile(String cacheFile) { this.cacheFile = cacheFile; }
    public String getSystemPrompt() { return systemPrompt; }
    public void setSystemPrompt(String systemPrompt) { this.systemPrompt = systemPrompt; }
    public Map<String, ModelConfig> getModels() { return models; }
    public void setModels(Map<String, ModelConfig> models) { this.models = models; }
    public Generation getGeneration() { return generation; }
    public void setGeneration(Generation generation) { this.generation = generation; }
}
