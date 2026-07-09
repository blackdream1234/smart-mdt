use smart_mdt_rs::data::BitSet;

fn bools_to_bitset(xs: &[bool]) -> BitSet {
    let mut b = BitSet::zeros(xs.len());
    for (i, x) in xs.iter().enumerate() {
        b.set(i, *x);
    }
    b
}

#[test]
fn bitset_operations_match_naive_vectors_for_partial_blocks() {
    for n in [0, 1, 2, 63, 64, 65, 127, 129] {
        let a: Vec<bool> = (0..n).map(|i| i % 3 == 0).collect();
        let c: Vec<bool> = (0..n).map(|i| i % 5 == 1).collect();
        let ba = bools_to_bitset(&a);
        let bc = bools_to_bitset(&c);
        assert_eq!(ba.count_ones(), a.iter().filter(|x| **x).count());
        let not_a = ba.not();
        assert_eq!(not_a.count_ones(), n - ba.count_ones());
        for i in 0..n {
            assert_eq!(ba.and(&bc).unwrap().get(i), a[i] & c[i]);
            assert_eq!(ba.or(&bc).unwrap().get(i), a[i] | c[i]);
            assert_eq!(ba.xor(&bc).unwrap().get(i), a[i] ^ c[i]);
            assert_eq!(not_a.get(i), !a[i]);
        }
    }
}

#[test]
fn bitset_dimension_mismatch_is_recoverable_error() {
    assert!(BitSet::zeros(3).and(&BitSet::zeros(4)).is_err());
}
