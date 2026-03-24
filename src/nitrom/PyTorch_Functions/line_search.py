import torch


class LineSearch:
    def __init__(
        self,
        c=0.1,
        tau=0.5,
        max_iter=20,
        alpha_init=1.0,
        *,
        min_alpha=1e-12,
        allow_increase_factor=1.01,
        reset_oldalpha_each_call=True,
    ):
        self.c = float(c)
        self.tau = float(tau)
        self.max_iter = int(max_iter)

        self.alpha_init = float(alpha_init)

        self.min_alpha = float(min_alpha)
        self.allow_increase_factor = (
            float(allow_increase_factor) if allow_increase_factor is not None else None
        )
        self.reset_oldalpha_each_call = bool(reset_oldalpha_each_call)

        self._oldalpha = None

    @staticmethod
    def _loss_to_float(loss):
        if torch.is_tensor(loss):
            return float(loss.detach().item())
        return float(loss)

    @staticmethod
    def _direction_norm(descent_direction):
        # Euclidean norm over a list of tensors
        with torch.no_grad():
            sq = None
            for d in descent_direction:
                if not torch.is_tensor(d):
                    d = torch.as_tensor(d)
                v = (d.detach() ** 2).sum()
                sq = v if sq is None else (sq + v)
            if sq is None:
                return 0.0
            return float(torch.sqrt(sq).item())

    def search(self, cost_fn, params, descent_direction, current_loss, grad_norm_sq):
        """
        Returns: (alpha, trial_loss, num_evals)
        """
        f0 = self._loss_to_float(current_loss)
        grad_norm_sq_f = float(grad_norm_sq)

        norm_d = self._direction_norm(descent_direction)
        if norm_d == 0.0:
            return 0.0, current_loss, 0

        # PyManopt: alpha = oldalpha else initial_step_size / norm(d)
        if (self._oldalpha is not None) and (not self.reset_oldalpha_each_call):
            alpha = float(self._oldalpha)
        else:
            alpha = float(self.alpha_init / norm_d)

        # Evaluate and backtrack (Armijo)
        num_evals_total = 0
        trial_loss = current_loss

        def _eval(alpha_local):
            trial_params = [
                p - alpha_local * d for p, d in zip(params, descent_direction)
            ]
            loss_val = cost_fn(trial_params)
            return loss_val

        # First trial
        trial_loss = _eval(alpha)
        num_evals_total += 1

        # Backtracking loop
        while (
            self._loss_to_float(trial_loss) > (f0 - self.c * alpha * grad_norm_sq_f)
            and num_evals_total < self.max_iter
        ):
            alpha *= self.tau
            trial_loss = _eval(alpha)
            num_evals_total += 1

        # Fallback (like your PyManopt modification): if alpha got too small, allow +1% increase
        if self.allow_increase_factor is not None and alpha <= self.min_alpha:
            print("Attention: allowing for cost function to increase by 1 percent")

            alpha = float(self.alpha_init / norm_d)
            self._oldalpha = alpha  # matches your intent, though it may be reset below

            trial_loss = _eval(alpha)
            num_evals_total += 1

            # backtrack until f(new) <= 1.01 * f0
            while (
                self._loss_to_float(trial_loss) > (self.allow_increase_factor * f0)
                and num_evals_total < 2 * self.max_iter
            ):
                alpha *= self.tau
                trial_loss = _eval(alpha)
                num_evals_total += 1

        # Suggest next alpha (disabled by default to mirror your pasted code)
        if self.reset_oldalpha_each_call:
            self._oldalpha = None
        else:
            # Roughly mimic your scaling logic
            if num_evals_total == 2:
                self._oldalpha = 10.0 * alpha
            else:
                self._oldalpha = 100.0 * alpha

        return alpha, trial_loss, num_evals_total
