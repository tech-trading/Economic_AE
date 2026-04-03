# STRATEGIES

Este documento describe en detalle cómo funciona cada estrategia implementada en `src/strategies.py`.

Entrada común a las estrategias
- `event_row` (pd.Series): fila de evento usada para contexto (puede incluir `date_utc`, `symbol`, etc.).
- `ticks` (pd.DataFrame): ticks recientes con columnas mínimas `time_utc`, `bid`, `ask`.
- `bundle`: objeto que contiene `X_tabular` (datos tabulares) y `X_seq` (secuencias para LSTM).
- `tabular_models`, `lstm_model`, `feature_columns`: modelos y columnas usados por `Default`, `ZScore` y `Momentum`.
- `policy` (dict): umbrales globales como `decision_threshold` y `no_trade_band`.
- `settings`: configuración de entorno / claves usadas para construir instancias.

Salida
- `TradeDecision` con campos: `side` ("BUY"/"SELL"), `confidence` (0..1), `proba_buy` (prob. del modelo).
- Las estrategias devuelven `None` cuando no hay señal válida o no se cumplen umbrales.

---

**DefaultStrategy**:
- Requisitos: `requires_models=True`, `requires_event=True`.
- Propósito: usar el ensamble de modelos (`ensemble_predict_proba`) para decidir BUY/SELL.
- Flujo:
  - Si `bundle.X_tabular` está vacío → no operar.
  - Construye `x_row` reindexado a `feature_columns` y llama a `ensemble_predict_proba(tabular_models, lstm_model, x_row, bundle.X_seq[0])` → `proba_buy`.
  - `side` = BUY si `proba_buy >= 0.5` else SELL.
  - `confidence` = max(proba_buy, 1-proba_buy).
  - Si `confidence < policy['decision_threshold']` o `abs(proba_buy-0.5) < policy['no_trade_band']` → no operar.
  - Retorna `TradeDecision(side, confidence, proba_buy)`.
- Parámetros relevantes: ninguno propio; depende de `policy`.

**ZScoreStrategy**:
- Requisitos: usa modelos + ticks.
- Parámetros (por defecto): `lookback_seconds=300`, `z_threshold=0.7`, `z_weight=1.0`, `mode='weighted'`.
- Cálculo de Z:
  - Toma ventana de ticks de los últimos `lookback_seconds` (usa `time_utc` si existe).
  - Calcula `mid = (bid+ask)/2`, su media y desviación estándar.
  - `z = (last_mid - mean_mid)/std_mid` (si std==0 → z=0).
- Modos de combinación:
  - `conjunctive`: exige que la dirección del modelo (`proba_buy>=0.5` → +1) coincida con la dirección del Z (Z > z_threshold → +1; Z < -z_threshold → -1). Si no coinciden o Z en banda → no operar. Si coinciden, aplica umbrales de `policy` y devuelve decisión basada en modelo.
  - `weighted` (por defecto): combina `model_score = proba_buy - 0.5` y `z_norm = tanh(z)` en: `combined = model_score + (z_weight * z_norm / 2.0)`. Determina `side` por signo de `combined`. `confidence = min(1.0, abs(combined))`. Aplica `policy` y `no_trade_band`.
- Casos que devuelven `None`: ventana vacía, confidence < threshold, dentro de `no_trade_band`.

**MomentumStrategy**:
- Requisitos: usa modelos + ticks.
- Parámetros (por defecto): `lookback_seconds=300`, `momentum_threshold=0.0005`, `momentum_weight=1.0`, `mode='weighted'`.
- Cálculo de momentum:
  - Selecciona ventana de `lookback_seconds` y calcula `mid` como (bid+ask)/2.
  - Momentum simple = (last - first) / first (proporción).
- Modos:
  - `conjunctive`: requiere que la dirección del modelo y la del momentum (exceder umbral positivo/negativo) coincidan; si no, no operar.
  - `weighted`: normaliza el momentum via `tanh(mom * 200.0)` y combina con `model_score` igual que en ZScore: `combined = model_score + (momentum_weight * mom_norm / 2.0)`. Calcula `confidence` y aplica `policy` y `no_trade_band`.
- Notas: el factor 200 en la normalización escala el momentum porcentual a un rango práctico antes de aplicar `tanh`.

**EmaRsiTrendStrategy**:
- Requisitos: `requires_models=False`, `requires_event=False` (no necesita modelos ni evento).
- Parámetros principales (valores por defecto mostrados en el constructor):
  - `fast_span=21`, `slow_span=55` — spans de EMA (slow > fast).
  - `rsi_period=14`, `rsi_buy_level=56.0`, `rsi_sell_level=44.0` — RSI y niveles.
  - `min_separation_pips=0.20` — separación mínima en pips entre EMAs requerida.
  - `momentum_lookback_ticks=20`, `min_momentum_pips=0.25` — momentum en pips y lookback.
  - `vol_period=40`, `min_vol_pips=0.05` — volatilidad mínima (media móvil de cambios absolutos).
- Flujo de decisión:
  - Ordena ticks por `time_utc` si es necesario y calcula `mid`.
  - Calcula `ema_fast` y `ema_slow` por EWM.
  - `ema_gap_pips = (ema_fast[-1] - ema_slow[-1]) / pip_size(symbol)`.
  - Calcula `rsi` usando EWM de ganancias/pérdidas (retorna ~50 si no hay suficientes datos).
  - Calcula `momentum_pips` sobre `momentum_lookback_ticks` (ultima - anterior) / pip.
  - Calcula `vol_pips` como media rodante de abs(diff(mid)) sobre `vol_period` en pips; si `vol_pips < min_vol_pips` → no operar.
  - `buy_ok` se cumple cuando: `ema_gap_pips >= min_separation_pips` AND `rsi >= rsi_buy_level` AND `momentum_pips >= min_momentum_pips`.
  - `sell_ok` análogo para señales bajistas.
  - Si ambas o ninguna son verdaderas → no operar.
  - Calcula `strength_gap`, `strength_mom`, `rsi_edge` y une en una fórmula de confianza: base 0.52 + ponderaciones; recorta entre [0.52,0.94]. Si `confidence < policy['decision_threshold']` → no operar.
  - Estima `proba_buy` a partir de `confidence` y `direction` y retorna `TradeDecision`.
- Observaciones:
  - Usa `symbol` desde `settings` para determinar `pip` (0.01 para JPY, 0.0001 para otros).
  - Buen candidato para reglas basadas en tendencia y ruido controlado (vol).

**AgenticHybridStrategy**:
- Requisitos: `requires_models=False`, `requires_event=False`.
- Propósito: agente meta que combina y aprende entre sub-agentes (`ema_rsi` y `donchian`) ajustando pesos por recompensa.
- Parámetros y estado:
  - `learning_rate` (por defecto desde `settings.agentic_learning_rate`, clip 0.01..1.0).
  - `explore_prob` (probabilidad de exploración aleatoria, por defecto ~0.10).
  - `min_agent_confidence` (umbral mínimo para aceptar decisión de un sub-agente, por defecto ~0.56).
  - `reward_horizon_seconds` (tiempo tras la entrada para evaluar la recompensa) y `reward_target_pips` (escala de recompensa).
  - `state_path` (ruta JSON para persistir `weights` y `agent_counts`).
  - `weights` iniciales para agentes: `{'ema_rsi':1.0, 'donchian':1.0}`.
  - `pending_trades`: lista de entradas pendientes para las cuales se evaluará recompensa cuando venza el `due_time`.
- Sub-agentes usados:
  - `EmaRsiTrendStrategy` (configurado con parámetros de `settings`).
  - `DonchianBreakoutStrategy` (configurado con parámetros de `settings`).
- Flujo:
  - En cada `decide` actualiza recompensas de `pending_trades` si su `due_time` ha pasado: calcula retorno en pips relativo a `entry_mid`, normaliza con `reward_target_pips` y actualiza peso `weights[agent] += learning_rate * reward` (clip entre 0.2 y 5.0). Persiste estado si cambia.
  - Pide decisiones a sub-agentes; incluye en `candidates` solo si `decision` no es None y `confidence >= min_agent_confidence`.
  - Si no hay candidatos → no operar.
  - Selección del agente:
    - Con probabilidad `explore_prob` escoge aleatoriamente (exploración).
    - Si no, normaliza pesos (`w_norm`) y calcula score = 0.70*w_norm + 0.25*confidence + 0.05*edge, donde `edge = abs(proba_buy-0.5)`.
    - Escoge el candidato con mayor score.
  - Verifica `policy['decision_threshold']` y si excede, añade la orden a `pending_trades` con `due_time=now+reward_horizon_seconds` y devuelve la `decision` seleccionada.
- Observaciones:
  - Es un meta-agente orientado a auto-ajustar qué estrategia produce mejores resultados en el horizonte de recompensa.
  - Estado persistente permite aprendizaje entre ejecuciones.

**DonchianBreakoutStrategy**:
- Requisitos: `requires_models=False`.
- Parámetros (por defecto visibles en constructor):
  - `lookback_seconds=600`, `breakout_buffer_pips=0.2`, `min_channel_pips=1.0`, `confirm_ticks=1`, `trigger_quantile=0.80`, `session_filter=False`, `sessions='london,ny'`.
- Flujo:
  - Construye ventana de ticks de `lookback_seconds` (o usa últimas N si no hay `time_utc`).
  - Si `session_filter` está activo, valida que la hora del evento esté dentro de ventanas London/NY.
  - Calcula `mid`, separa `pivot = mid[:-confirm_ticks]` y `latest_block = last confirm_ticks`.
  - `high = pivot.max()`, `low = pivot.min()`; `channel_width = high-low` y `channel_pips = channel_width / pip`.
  - Si `channel_pips < min_channel_pips` → no operar (canal demasiado estrecho).
  - `buffer = breakout_buffer_pips * pip`.
  - `buy_break` si todos los ticks de `latest_block` > `high + buffer`.
  - `sell_break` si todos los ticks de `latest_block` < `low - buffer`.
  - Calcula `channel_pos = (latest - low) / channel_width` y define `buy_zone`/`sell_zone` según `trigger_quantile`; si no hay confirmación de bloque, usa zona como alternativa.
  - Si ambas o ninguna condiciones de ruptura se cumplen → no operar.
  - Calcula `ema_fast` (span=20) y `ema_slow` (span=50) para modular `trend_factor` (si ruptura contraria a EMA, reduce un poco la confianza).
  - `strength = distance / channel_width`, `edge_strength = max(strength, abs(channel_pos-0.5)*2)` y `confidence = clip((0.55 + min(0.35, edge_strength*0.45)) * trend_factor, 0.55, 0.93)`.
  - Si `confidence < policy['decision_threshold']` → no operar.
  - `proba_buy = clip(0.5 + direction * min(0.49, 0.28 + strength), 0.01, 0.99)` y retorna `TradeDecision`.
- Observaciones: es una estrategia clásica de ruptura de canal (Donchian/Turtle), con buffer y confirmación de ticks.

---

`get_strategy(name, settings, policy)`
- Mapea nombres a constructores e inyecta parámetros desde `settings`. Nombres admitidos (ejemplos):
  - `zscore`, `momentum`, `donchian`, `donchian_nylondon`, `ema_rsi`, `agentic`, etc.
- Si no reconoce el nombre, retorna `DefaultStrategy()`.

---

Sugerencias prácticas
- Ajustar `policy['decision_threshold']` y `policy['no_trade_band']` para controlar la aversión al riesgo y evitar señales débiles.
- Para `AgenticHybridStrategy` revisar `agentic_state.json` (ruta `settings.agentic_state_path`) para entender cómo evolucionan los pesos.
- Probar `EmaRsiTrendStrategy` con datos ordenados por `time_utc` y suficientes ticks (>= slow_span + vol_period + margen) para evitar retornos `None` por falta de datos.

Si quieres, puedo:
- Añadir ejemplos concretos de configuración en `settings` y `policy`.
- Generar tests unitarios simples para cada estrategia.
