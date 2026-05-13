package kz.kazgpt.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import kz.kazgpt.config.KazGptProperties;
import kz.kazgpt.model.CachedResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import jakarta.annotation.PostConstruct;
import java.io.InputStream;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

@Service
public class CacheService {

    private static final Logger log = LoggerFactory.getLogger(CacheService.class);

    private final KazGptProperties props;
    private final ResourceLoader resourceLoader;
    private final ObjectMapper mapper = new ObjectMapper();
    private List<CachedResponse> cache = List.of();

    public CacheService(KazGptProperties props, ResourceLoader resourceLoader) {
        this.props = props;
        this.resourceLoader = resourceLoader;
    }

    @PostConstruct
    public void load() {
        try {
            Resource res = resourceLoader.getResource(props.getCacheFile());
            if (!res.exists()) {
                log.warn("Cache file not found: {}", props.getCacheFile());
                return;
            }
            try (InputStream in = res.getInputStream()) {
                CachedResponse[] arr = mapper.readValue(in, CachedResponse[].class);
                cache = Arrays.asList(arr);
                log.info("Loaded {} cached responses", cache.size());
            }
        } catch (Exception e) {
            log.error("Failed to load cache: {}", e.getMessage());
        }
    }

    public Optional<CachedResponse> findMatch(String userMessage) {
        if (cache.isEmpty()) return Optional.empty();
        String normalized = normalize(userMessage);

        for (CachedResponse cr : cache) {
            String cn = normalize(cr.question());
            if (cn.equals(normalized) || cn.startsWith(normalized) || normalized.startsWith(cn)) {
                return Optional.of(cr);
            }
        }

        int bestDistance = Integer.MAX_VALUE;
        CachedResponse best = null;
        for (CachedResponse cr : cache) {
            int d = levenshtein(normalize(cr.question()), normalized);
            if (d < bestDistance) {
                bestDistance = d;
                best = cr;
            }
        }

        int threshold = Math.max(5, normalized.length() / 4);
        if (best != null && bestDistance <= threshold) {
            return Optional.of(best);
        }

        return Optional.empty();
    }

    public List<CachedResponse> all() {
        return cache;
    }

    private static String normalize(String s) {
        return s == null ? "" : s.trim().toLowerCase().replaceAll("[?!.,\\s]+", " ").trim();
    }

    private static int levenshtein(String a, String b) {
        int[][] d = new int[a.length() + 1][b.length() + 1];
        for (int i = 0; i <= a.length(); i++) d[i][0] = i;
        for (int j = 0; j <= b.length(); j++) d[0][j] = j;
        for (int i = 1; i <= a.length(); i++) {
            for (int j = 1; j <= b.length(); j++) {
                int cost = a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1;
                d[i][j] = Math.min(Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1), d[i - 1][j - 1] + cost);
            }
        }
        return d[a.length()][b.length()];
    }
}
