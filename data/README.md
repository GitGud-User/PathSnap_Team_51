# data/ (not tracked by git)

Copy the supplied dataset here so that it looks like this:

```
data/
├── train/<trip_id>/<trip_id>.{track,nodes,arcs,route}          80 trips
├── student_test/<trip_id>/<trip_id>.{track,nodes,arcs,route}   10 trips
└── evaluator_test/                                             filled only on Laptop 1 on evaluation day
```

Or point `PATHSNAP_DATA_DIR` at another folder with the same layout.
Never commit data, and never commit evaluator files in particular.
