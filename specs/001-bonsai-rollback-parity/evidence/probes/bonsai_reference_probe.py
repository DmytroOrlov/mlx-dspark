"""Interrupted reference-probe marker; no executable probe was recovered.

The attempted write to /private/tmp/bonsai_reference_probe.py was interrupted
before a file existed. Resume T004/T005 by implementing the reference probe here.
"""

raise RuntimeError("Reference probe was interrupted before creation; resume T004/T005")
