# Adversarial design audit: Mixed-Kill multi-seed replication

Seeds 12 and 13 reproduce the original 2,000-episode Mixed-Kill protocol from
the same baseline checkpoint. Only the training seed changes; both use the same
features, rewards, optimizer, replay, epsilon and peaceful + two rule-based
opponents. Each gets visible Quick100 checks in 3RB and External Trio.

This estimates training variance. It is not permission to select a fortunate
seed from Quick100: a challenger needs independent Full1000 confirmation in
both fields before it can replace the seed-11 incumbent. The two arms have
separate artifacts and may run concurrently.
