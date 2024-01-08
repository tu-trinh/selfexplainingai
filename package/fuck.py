import pkg_resources

dist = pkg_resources.get_distribution('gymnasium')
if dist.has_metadata('METADATA'):
    metadata = dist.get_metadata('METADATA')
elif dist.has_metadata('PKG-INFO'):
    metadata = dist.get_metadata('PKG-INFO')

# Now parse 'metadata' for 'Provides-Extra' entries
for line in metadata.splitlines():
    if line.startswith('Provides-Extra:'):
        extra = line.split(':', 1)[1].strip()
        print(extra)