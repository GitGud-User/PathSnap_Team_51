// Stage-1 dataset audit (C#, because no Python is installed on this machine yet).
// Reads every trip folder and writes one CSV row per trip. Ground-truth .route is only
// used here for *auditing* (route validity, GPS-to-route distance), never for prediction.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

public static class PathSnapAudit
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;
    const double R = 6371008.8;

    static double Hav(double lon1, double lat1, double lon2, double lat2)
    {
        double p1 = lat1 * Math.PI / 180, p2 = lat2 * Math.PI / 180;
        double dp = p2 - p1, dl = (lon2 - lon1) * Math.PI / 180;
        double a = Math.Sin(dp / 2) * Math.Sin(dp / 2) + Math.Cos(p1) * Math.Cos(p2) * Math.Sin(dl / 2) * Math.Sin(dl / 2);
        return 2 * R * Math.Asin(Math.Min(1, Math.Sqrt(a)));
    }

    // point-to-segment distance in a local equirectangular projection (metres)
    static double PtSeg(double px, double py, double ax, double ay, double bx, double by, double lat0)
    {
        double kx = Math.Cos(lat0 * Math.PI / 180) * Math.PI / 180 * R, ky = Math.PI / 180 * R;
        double X = (px - ax) * kx, Y = (py - ay) * ky, BX = (bx - ax) * kx, BY = (by - ay) * ky;
        double L2 = BX * BX + BY * BY;
        double t = L2 > 0 ? Math.Max(0, Math.Min(1, (X * BX + Y * BY) / L2)) : 0;
        double dx = X - t * BX, dy = Y - t * BY;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    static int Find(int[] p, int x) { while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; } return x; }

    static double Pct(List<double> v, double q)
    {
        if (v.Count == 0) return double.NaN;
        var s = v.OrderBy(x => x).ToList();
        return s[(int)Math.Min(s.Count - 1, Math.Floor(q * (s.Count - 1)))];
    }

    public static string Run(string root, string outCsv)
    {
        var sb = new StringBuilder();
        sb.AppendLine(string.Join(",", new[]{"partition","trip","files","track_rows","track_bad_rows","t_start","t_end","duration_s",
            "dt_eq1_pct","dt_gt1_count","dt_max","dt_nonpos_count","dup_coord_steps","gps_path_m","speed_p99_mps","speed_max_mps","jumps_gt50mps",
            "lon_min","lon_max","lat_min","lat_max","nodes","node_bad_rows","arcs","arc_bad_rows","arc_bad_ref","self_loops","dup_arcs",
            "reverse_pair_pct","wcc_count","largest_wcc_nodes_pct","isolated_nodes","arc_len_median_m","arc_len_p95_m","arc_len_max_m","zero_len_arcs",
            "route_len","route_bad_ref","route_consec_dup","route_breaks","route_len_m","route_unique_arcs",
            "gps2route_median_m","gps2route_p95_m","gps2route_max_m","gps_gt50m_from_route","net_bbox_km2","track_in_net_bbox_pct"}));

        foreach (var part in new[] { "train", "student_test" })
        {
            var pdir = Path.Combine(root, part);
            if (!Directory.Exists(pdir)) continue;
            foreach (var tdir in Directory.GetDirectories(pdir).OrderBy(d => d))
            {
                string id = Path.GetFileName(tdir);
                var files = Directory.GetFiles(tdir).Select(Path.GetFileName).OrderBy(x => x).ToList();
                Func<string, string> F = ext => Path.Combine(tdir, id + ext);

                // ---- track
                var tl = new List<double>(); var ta = new List<double>(); var tt = new List<double>(); int tbad = 0;
                if (File.Exists(F(".track")))
                    foreach (var line in File.ReadLines(F(".track")))
                    {
                        var p = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        double a, b, c;
                        if (p.Length != 3 || !double.TryParse(p[0], NumberStyles.Float, C, out a) || !double.TryParse(p[1], NumberStyles.Float, C, out b) || !double.TryParse(p[2], NumberStyles.Float, C, out c)
                            || double.IsNaN(a) || double.IsNaN(b) || Math.Abs(a) > 180 || Math.Abs(b) > 90) { if (line.Trim().Length > 0) tbad++; continue; }
                        tl.Add(a); ta.Add(b); tt.Add(c);
                    }
                int n = tl.Count; int dt1 = 0, dtgt1 = 0, dtnp = 0, dupc = 0, jumps = 0; double dtmax = 0, path = 0; var spd = new List<double>();
                for (int i = 1; i < n; i++)
                {
                    double dt = tt[i] - tt[i - 1];
                    if (Math.Abs(dt - 1) < 1e-6) dt1++; else if (dt > 1) dtgt1++;
                    if (dt <= 0) dtnp++;
                    dtmax = Math.Max(dtmax, dt);
                    double d = Hav(tl[i - 1], ta[i - 1], tl[i], ta[i]); path += d;
                    if (d == 0) dupc++;
                    if (dt > 0) { double s = d / dt; spd.Add(s); if (s > 50) jumps++; }
                }

                // ---- nodes
                var nl = new List<double>(); var na = new List<double>(); int nbad = 0;
                if (File.Exists(F(".nodes")))
                    foreach (var line in File.ReadLines(F(".nodes")))
                    {
                        var p = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        double a, b;
                        if (p.Length != 2 || !double.TryParse(p[0], NumberStyles.Float, C, out a) || !double.TryParse(p[1], NumberStyles.Float, C, out b)) { nbad++; nl.Add(double.NaN); na.Add(double.NaN); continue; }
                        nl.Add(a); na.Add(b);
                    }
                int N = nl.Count;

                // ---- arcs
                var af = new List<int>(); var at = new List<int>(); int abad = 0, aref = 0, self = 0;
                if (File.Exists(F(".arcs")))
                    foreach (var line in File.ReadLines(F(".arcs")))
                    {
                        var p = line.Split(new[] { ' ', '\t' }, StringSplitOptions.RemoveEmptyEntries);
                        int u, v;
                        if (p.Length != 2 || !int.TryParse(p[0], out u) || !int.TryParse(p[1], out v)) { abad++; af.Add(-1); at.Add(-1); continue; }
                        if (u < 0 || v < 0 || u >= N || v >= N) aref++;
                        if (u == v) self++;
                        af.Add(u); at.Add(v);
                    }
                int A = af.Count;
                var set = new HashSet<long>(); int dup = 0, rev = 0;
                for (int i = 0; i < A; i++) { long k = (long)af[i] * 10000000L + at[i]; if (!set.Add(k)) dup++; }
                for (int i = 0; i < A; i++) if (set.Contains((long)at[i] * 10000000L + af[i])) rev++;
                var par = Enumerable.Range(0, N).ToArray(); var deg = new int[N];
                var alen = new List<double>(); int zl = 0;
                for (int i = 0; i < A; i++)
                {
                    int u = af[i], v = at[i]; if (u < 0 || v < 0 || u >= N || v >= N) { alen.Add(double.NaN); continue; }
                    deg[u]++; deg[v]++; int ru = Find(par, u), rv = Find(par, v); if (ru != rv) par[ru] = rv;
                    double L = Hav(nl[u], na[u], nl[v], na[v]); alen.Add(L); if (L < 0.01) zl++;
                }
                var comp = new Dictionary<int, int>(); int iso = 0;
                for (int i = 0; i < N; i++) { int r = Find(par, i); comp[r] = comp.ContainsKey(r) ? comp[r] + 1 : 1; if (deg[i] == 0) iso++; }
                int largest = comp.Count > 0 ? comp.Values.Max() : 0;
                var alenOk = alen.Where(x => !double.IsNaN(x)).ToList();

                // ---- route (audit only)
                var route = new List<int>(); int rbad = 0;
                if (File.Exists(F(".route")))
                    foreach (var line in File.ReadLines(F(".route")))
                    {
                        int r; if (int.TryParse(line.Trim(), out r)) route.Add(r); else if (line.Trim().Length > 0) rbad++;
                    }
                int rref = 0, rdup = 0, rbreak = 0; double rlen = 0;
                for (int i = 0; i < route.Count; i++)
                {
                    int r = route[i]; if (r < 0 || r >= A) { rref++; continue; }
                    if (!double.IsNaN(alen[r])) rlen += alen[r];
                    if (i > 0) { int q = route[i - 1]; if (q == r) rdup++; else if (q >= 0 && q < A && at[q] != af[r]) rbreak++; }
                }
                // GPS distance to nearest reference-route arc
                var g2r = new List<double>(); int far = 0;
                var ra = route.Where(r => r >= 0 && r < A && af[r] >= 0 && af[r] < N && at[r] >= 0 && at[r] < N).Distinct().ToArray();
                if (ra.Length > 0)
                    for (int i = 0; i < n; i++)
                    {
                        double best = double.MaxValue;
                        foreach (var r in ra) { double d = PtSeg(tl[i], ta[i], nl[af[r]], na[af[r]], nl[at[r]], na[at[r]], ta[i]); if (d < best) best = d; }
                        g2r.Add(best); if (best > 50) far++;
                    }

                double nlmin = nl.Where(x => !double.IsNaN(x)).DefaultIfEmpty(double.NaN).Min(), nlmax = nl.Where(x => !double.IsNaN(x)).DefaultIfEmpty(double.NaN).Max();
                double namin = na.Where(x => !double.IsNaN(x)).DefaultIfEmpty(double.NaN).Min(), namax = na.Where(x => !double.IsNaN(x)).DefaultIfEmpty(double.NaN).Max();
                double bbox = Hav(nlmin, namin, nlmax, namin) / 1000 * Hav(nlmin, namin, nlmin, namax) / 1000;
                int inb = 0; for (int i = 0; i < n; i++) if (tl[i] >= nlmin && tl[i] <= nlmax && ta[i] >= namin && ta[i] <= namax) inb++;

                Func<double, string> f = x => double.IsNaN(x) ? "" : x.ToString("0.######", C);
                sb.AppendLine(string.Join(",", new[]{part, id, string.Join("|", files.Select(x=>Path.GetExtension(x))), n.ToString(), tbad.ToString(),
                    n>0?f(tt[0]):"", n>0?f(tt[n-1]):"", n>0?f(tt[n-1]-tt[0]):"",
                    n>1?f(100.0*dt1/(n-1)):"", dtgt1.ToString(), f(dtmax), dtnp.ToString(), dupc.ToString(), f(path), f(Pct(spd,0.99)), spd.Count>0?f(spd.Max()):"", jumps.ToString(),
                    n>0?f(tl.Min()):"", n>0?f(tl.Max()):"", n>0?f(ta.Min()):"", n>0?f(ta.Max()):"",
                    N.ToString(), nbad.ToString(), A.ToString(), abad.ToString(), aref.ToString(), self.ToString(), dup.ToString(),
                    A>0?f(100.0*rev/A):"", comp.Count.ToString(), N>0?f(100.0*largest/N):"", iso.ToString(),
                    f(Pct(alenOk,0.5)), f(Pct(alenOk,0.95)), alenOk.Count>0?f(alenOk.Max()):"", zl.ToString(),
                    route.Count.ToString(), rref.ToString(), rdup.ToString(), rbreak.ToString(), f(rlen), route.Distinct().Count().ToString(),
                    f(Pct(g2r,0.5)), f(Pct(g2r,0.95)), g2r.Count>0?f(g2r.Max()):"", far.ToString(), f(bbox), n>0?f(100.0*inb/n):""}));
            }
        }
        File.WriteAllText(outCsv, sb.ToString());
        return outCsv;
    }
}
