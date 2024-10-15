import hashlib

hs = hashlib.md5(repr(run_ids).encode("utf-8")).hexdigest()
filename = f"parcoords_{hs}.pdf"

print(f"Output to {filename}")
fig.write_image(filename)
