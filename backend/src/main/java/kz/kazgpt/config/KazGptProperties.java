package kz.kazgpt.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.stereotype.Component;

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
        private double temperature = 0.3;
        private double topP = 0.8;
        private double repeatPenalty = 1.2;
        private int maxTokens = 512;

        public double getTemperature() { return temperature; }
        public void setTemperature(double temperature) { this.temperature = temperature; }
        public double getTopP() { return topP; }
        public void setTopP(double topP) { this.topP = topP; }
        public double getRepeatPenalty() { return repeatPenalty; }
        public void setRepeatPenalty(double repeatPenalty) { this.repeatPenalty = repeatPenalty; }
        public int getMaxTokens() { return maxTokens; }
        public void setMaxTokens(int maxTokens) { this.maxTokens = maxTokens; }
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
