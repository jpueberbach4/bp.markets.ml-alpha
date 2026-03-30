class TradeManager:
    """Manages lifecycle of trades including execution, evaluation, and performance tracking.

    This class simulates a trading engine with the following characteristics:
    - Applies pessimistic slippage on entries and stop-loss exits
    - Supports multi-target scaling via risk-reward (RR) ladders
    - Evaluates trades using candle high/low extremes (worst-case assumption)
    - Tracks performance metrics in R-multiples
    - Applies trailing stop logic based on achieved RR milestones
    """

    def __init__(self, rr_ratios, slippage_fraction=0.025):
        """Initializes the TradeManager.

        Args:
            rr_ratios (list[float]): List of risk-reward targets (e.g., [1, 2, 3, ...]).
            slippage_fraction (float, optional): Fraction of ATR used as slippage penalty.
                Defaults to 0.025.
        """
        # Risk-reward ladder defining multiple take-profit targets per trade
        self.rr_ratios = rr_ratios

        # Fraction of ATR used to simulate execution slippage
        self.slippage_fraction = slippage_fraction

        # Active trades currently being tracked
        self.open_trades = []

        # Performance metrics tracked in R-multiples
        self.metrics = {
            'wins': 0,
            'losses': 0,
            'breakevens': 0,
            'gross_profit': 0.0,
            'gross_loss': 0.0
        }

    def execute_pending(self, signal_price, cohort_id, entry_t):
        """Executes a pending trade signal and creates RR-scaled positions.

        A single signal results in multiple trades (one per RR level),
        effectively creating a ladder of take-profit targets.

        Entry price is adjusted pessimistically using slippage.

        Args:
            signal_price (float): Price at which the signal was generated.
            cohort_id (int): Identifier linking trades from the same signal.
            entry_t (int): Time index at which the trade is entered.

        Returns:
            None
        """
        # Approximate ATR as 1% of price (proxy for volatility)
        atr = signal_price * 0.01

        # Compute slippage based on ATR
        slippage = atr * self.slippage_fraction

        # Apply pessimistic fill (worse than signal price)
        fill_price = signal_price + slippage

        # Create one trade per RR level
        for rr in self.rr_ratios:
            self.open_trades.append({
                'cohort_id': cohort_id,
                'entry_t': entry_t,
                'entry': fill_price,

                # Stop-loss placed 1 ATR below signal price
                'sl': signal_price - atr,

                # Take-profit scaled by RR multiple
                'tp': signal_price + (atr * rr),

                # Store ATR and slippage for later calculations
                'atr': atr,
                'slippage': slippage,
                'rr': rr
            })

    def process_candle(self, c_high, c_low):
        """Evaluates all open trades against a new candle.

        Uses pessimistic execution:
        - Stop-loss is checked first (worst-case assumption)
        - If both SL and TP are hit within the same candle, SL wins

        Args:
            c_high (float): High price of the current candle.
            c_low (float): Low price of the current candle.

        Returns:
            list[dict]: List of closed trade records with performance metrics.
        """
        closed_trades = []

        # Iterate in reverse to safely remove trades while iterating
        for i in range(len(self.open_trades) - 1, -1, -1):
            tr = self.open_trades[i]

            # Initialize result variables
            res = 0
            r_multiple = 0.0
            is_breakeven = False

            # ======================================================
            # PESSIMISTIC EXECUTION: STOP-LOSS FIRST
            # ======================================================

            if c_low <= tr['sl']:
                # Apply additional slippage on stop-loss exit
                exit_price = tr['sl'] - tr['slippage']

                # Calculate raw PnL
                res = exit_price - tr['entry']

                # Convert to R-multiple
                r_multiple = res / tr['atr']
                
                # Deterministically flag breakevens based on exact stop location 
                is_breakeven = (tr['sl'] == tr['entry'])

                # Update performance metrics
                self._update_metrics(r_multiple, tr)

            # ======================================================
            # TAKE-PROFIT CONDITION
            # ======================================================

            elif c_high >= tr['tp']:
                # Exit at take-profit price (no extra slippage assumed here)
                exit_price = tr['tp']

                res = exit_price - tr['entry']
                r_multiple = res / tr['atr']

                # Update metrics for winning trade
                self._update_metrics(r_multiple, tr)

                # Apply trailing stop adjustments to remaining trades
                self._apply_trailing_stops(tr['cohort_id'], tr['rr'])

            # ======================================================
            # CLOSE TRADE IF EXIT CONDITION TRIGGERED
            # ======================================================

            if res != 0:
                closed_trades.append({
                    'cohort_id': tr['cohort_id'],
                    'r_multiple': r_multiple,
                    'is_breakeven': is_breakeven,
                    'entry_t': tr['entry_t'],
                    'entry': tr['entry'],
                    'exit_price': exit_price
                })

                # Remove trade from active list
                self.open_trades.pop(i)

        return closed_trades

    def _update_metrics(self, r_multiple, tr):
        """Updates performance metrics based on trade outcome.

        Args:
            r_multiple (float): Profit/loss expressed in R units.
            tr (dict): Trade dictionary for additional context.

        Returns:
            None
        """
        if r_multiple > 0:
            # Winning trade
            self.metrics['wins'] += 1
            self.metrics['gross_profit'] += r_multiple

        elif r_multiple == 0 or tr['sl'] == tr['entry']:
            # Breakeven trade (either exact or SL moved to entry)
            self.metrics['breakevens'] += 1

        else:
            # Losing trade
            self.metrics['losses'] += 1
            self.metrics['gross_loss'] += abs(r_multiple)

    def _apply_trailing_stops(self, cohort_id, triggered_rr):
        """Applies trailing stop logic based on achieved RR milestones.

        This function adjusts stop-loss levels of remaining trades within the same cohort
        when certain profit thresholds are reached.

        Example logic:
        - When RR=2 is hit → move SL to breakeven for RR>=3 trades
        - When RR=3 is hit → move SL further into profit for RR>=4 trades

        Args:
            cohort_id (int): Identifier for grouping related trades.
            triggered_rr (float): The RR level that was just achieved.

        Returns:
            None
        """
        # Move stop-loss to breakeven for higher RR trades
        if triggered_rr == 2.0:
            for ot in self.open_trades:
                if ot['cohort_id'] == cohort_id and ot['rr'] >= 3.0:
                    ot['sl'] = ot['entry']

        # Move stop-loss into profit (lock in gains)
        elif triggered_rr == 3.0:
            for ot in self.open_trades:
                if ot['cohort_id'] == cohort_id and ot['rr'] >= 4.0:
                    ot['sl'] = ot['entry'] + ot['atr'] + ot['slippage']