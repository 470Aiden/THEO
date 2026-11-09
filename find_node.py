# =============================================================================
# THEO - find the node on whatever network you are currently on
#
# WHY YOU NEED THIS:
#   main.py has NODE_IP = "192.168.0.18" written into it. That address only
#   means anything on your home network. On a phone hotspot the whole range
#   changes - Android hands out 192.168.43.x, iPhone hands out 172.20.10.x -
#   so a fixed address can never be right on both.
#
#   This works out what range YOU are on right now, then asks every address
#   on it whether it is the node. No hardcoding, no guessing.
#
# HOW TO USE IT:
#   Save this next to main.py as find_node.py, then near the top of main.py:
#
#       from find_node import find_node
#
#   and wherever you currently rely on NODE_IP, do this once at startup:
#
#       NODE_IP = find_node(known=("192.168.0.18",)) or "192.168.0.18"
#
#   Pass any addresses you already suspect in "known" - they are tried first
#   and answer instantly, so the scan only runs when they all fail.
# =============================================================================

import os
import socket
import requests
from concurrent.futures import ThreadPoolExecutor

# How long to give each address before moving on. Keep this small - the whole
# range is probed at once, so this is roughly how long the scan takes in total.
PROBE_TIMEOUT = 0.4

# How many addresses to probe at the same time.
PROBE_WORKERS = 64

# The node's /status reply announces itself with this. It is what tells the
# node apart from anything else that happens to answer on port 80.
NODE_FINGERPRINT = "THEO-ESP32"

_cached = None


def own_address():
    """
    Work out this machine's own address on the current network.

    Nothing is actually sent - opening a UDP socket toward a public address
    is just how you ask the operating system which interface it would use.
    """

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        probe.connect(("8.8.8.8", 80))
        return probe.getsockname()[0]
    finally:
        probe.close()


def is_container_network(address):
    """
    Is this address a container's private network rather than your Wi-Fi?

    When the app runs inside a container, the address it sees is handed out
    by the container system, not by your router. Those live in 172.17.x.x
    through 172.31.x.x. Your node, sitting on your home Wi-Fi, can never be
    on one of them - so sweeping that range finds nothing, every time.
    """

    parts = address.split(".")

    if len(parts) != 4 or parts[0] != "172" or not parts[1].isdigit():
        return False

    return 17 <= int(parts[1]) <= 31


def prefix_of(address):
    """192.168.0.18 -> 192.168.0"""

    parts = address.split(".")

    if len(parts) != 4:
        return None

    return ".".join(parts[:3])


def networks_worth_sweeping(known=(), subnets=()):
    """
    Work out which networks are actually worth looking on, in order.

    The strongest clue is the address you already had written down: if the
    node used to be at 192.168.0.18, then 192.168.0 is a real network in
    this house, whatever this machine happens to think it is on.
    """

    found = []

    def add(prefix):
        if prefix and prefix not in found:
            found.append(prefix)

    # 1. Anything passed in directly.
    for entry in subnets:
        add(prefix_of(entry) if entry.count(".") == 3 else entry)

    # 2. Anything set in the environment, e.g. THEO_NODE_SUBNETS=192.168.0
    for entry in os.environ.get("THEO_NODE_SUBNETS", "").replace(",", " ").split():
        add(prefix_of(entry) if entry.count(".") == 3 else entry)

    # 3. The networks the addresses you already know live on.
    for entry in known:
        add(prefix_of(entry))

    # 4. This machine's own network - unless that network is a container's
    #    private one, in which case the node can never be on it and looking
    #    there is twelve hundred wasted requests every single startup.
    mine = None

    try:
        mine = own_address()
    except Exception as error:
        print("[node] could not work out this network: " + str(error))

    if mine and is_container_network(mine):
        print("[node] " + mine + " is a container network, not your Wi-Fi")
        print("[node] nothing on your Wi-Fi can be there, so I will not look")

        for prefix in ("192.168.0", "192.168.1", "192.168.100", "10.0.0"):
            add(prefix)
    elif mine:
        add(prefix_of(mine))

    return found, mine


def is_the_node(address):
    """
    Ask one address whether it is actually the node. Returns it, or None.

    A 200 on its own is NOT enough - a router page, a printer, or something
    running locally can answer 200 and you would latch onto the wrong box.
    Your node's /status reply contains "THEO-ESP32", so that is what we
    check for. Nothing else on your network says that.
    """

    try:
        reply = requests.get(
            "http://" + address + "/status",
            timeout=PROBE_TIMEOUT
        )

        if reply.status_code == 200 and NODE_FINGERPRINT in reply.text:
            return address
    except Exception:
        pass

    return None


def find_node(known=(), use_cache=True, subnets=()):
    """
    Return the node's current address, or None if it is not on this network.

    Tries the addresses in "known" first, then sweeps the range this machine
    is sitting on.
    """

    global _cached

    if use_cache and _cached and is_the_node(_cached):
        return _cached

    # The addresses you already know about, asked first - they answer
    # instantly when they are right.
    if known:
        with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
            for found in pool.map(is_the_node, list(known)):
                if found:
                    print("[node] found at " + found)
                    _cached = found
                    return found

    prefixes, mine = networks_worth_sweeping(known=known, subnets=subnets)

    if not prefixes:
        print("[node] no network to search")
        return None

    for prefix in prefixes:
        addresses = [
            prefix + "." + str(last)
            for last in range(1, 255)
            if prefix + "." + str(last) != mine
        ]

        print("[node] searching " + prefix + ".1 to " + prefix + ".254")

        with ThreadPoolExecutor(max_workers=PROBE_WORKERS) as pool:
            for found in pool.map(is_the_node, addresses):
                if found:
                    print("[node] found at " + found)
                    _cached = found
                    return found

    print("[node] nothing on any of those networks answered like the node")
    return None
