use super::Cnf;
fn idx(l: i32) -> usize {
    let v = (l.abs() as usize) - 1;
    2 * v + if l > 0 { 0 } else { 1 }
}
fn neg_idx(i: usize) -> usize {
    i ^ 1
}
/// Checks 2-SAT using implication graph and SCC. Unit clauses are supported.
pub fn two_sat(num_vars: usize, cnf: &Cnf) -> bool {
    let n = 2 * num_vars;
    let mut g = vec![Vec::new(); n];
    let mut gr = vec![Vec::new(); n];
    for cl in cnf {
        if cl.is_empty() {
            return false;
        }
        let pairs: Vec<(i32, i32)> = if cl.len() == 1 {
            vec![(cl[0], cl[0])]
        } else if cl.len() == 2 {
            vec![(cl[0], cl[1])]
        } else {
            return false;
        };
        for (a, b) in pairs {
            let ia = idx(a);
            let ib = idx(b);
            for (u, v) in [(neg_idx(ia), ib), (neg_idx(ib), ia)] {
                g[u].push(v);
                gr[v].push(u);
            }
        }
    }
    let mut used = vec![false; n];
    let mut order = Vec::new();
    fn dfs(v: usize, g: &[Vec<usize>], used: &mut [bool], order: &mut Vec<usize>) {
        if used[v] {
            return;
        }
        used[v] = true;
        for &to in &g[v] {
            dfs(to, g, used, order)
        }
        order.push(v);
    }
    for v in 0..n {
        dfs(v, &g, &mut used, &mut order);
    }
    let mut comp = vec![usize::MAX; n];
    fn rdfs(v: usize, c: usize, gr: &[Vec<usize>], comp: &mut [usize]) {
        if comp[v] != usize::MAX {
            return;
        }
        comp[v] = c;
        for &to in &gr[v] {
            rdfs(to, c, gr, comp)
        }
    }
    for (c, &v) in order.iter().rev().enumerate() {
        rdfs(v, c, &gr, &mut comp);
    }
    (0..num_vars).all(|v| comp[2 * v] != comp[2 * v + 1])
}
