# Known Limitations

- Large artifacts are local and are not stored in git.
- A fresh clone can run schema checks and smoke tests, but full heavy
  reproduction requires regenerating or obtaining local artifacts.
- Legacy notebooks and experiment scripts are retained for research history and
  are not guaranteed to be the canonical reproduction path.
- Full trie-constrained generation over all users is slow compared with the
  lightweight smoke tests.
