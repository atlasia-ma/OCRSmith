"""Contract for the geometry primitives every annotation is built from."""

from dataclasses import FrozenInstanceError

import pytest

from ocrsmith.domain import BBox, Polygon


class TestBBoxConstruction:
    def test_coordinates_are_ordered_on_construction(self):
        assert BBox(30, 40, 10, 20) == BBox(10, 20, 30, 40)

    def test_dimensions(self):
        box = BBox(10, 20, 40, 60)

        assert (box.width, box.height, box.area) == (30, 40, 1200)

    def test_from_xywh(self):
        assert BBox.from_xywh(10, 20, 30, 40) == BBox(10, 20, 40, 60)

    def test_from_points(self):
        assert BBox.from_points([(5, 9), (1, 2), (3, 30)]) == BBox(1, 2, 5, 30)

    def test_from_points_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one point"):
            BBox.from_points([])

    def test_is_immutable(self):
        with pytest.raises(FrozenInstanceError):
            BBox(0, 0, 1, 1).x0 = 5

    def test_is_hashable(self):
        assert len({BBox(0, 0, 1, 1), BBox(0, 0, 1, 1)}) == 1


class TestBBoxTransforms:
    def test_translate(self):
        assert BBox(0, 0, 10, 10).translate(5, -2) == BBox(5, -2, 15, 8)

    def test_scale_uniformly(self):
        assert BBox(1, 2, 3, 4).scale(2) == BBox(2, 4, 6, 8)

    def test_scale_per_axis(self):
        assert BBox(1, 2, 3, 4).scale(2, 3) == BBox(2, 6, 6, 12)

    def test_clip_to_page(self):
        assert BBox(-5, -5, 50, 50).clip(0, 0, 20, 30) == BBox(0, 0, 20, 30)

    def test_clip_keeps_contained_boxes(self):
        box = BBox(2, 2, 8, 8)
        assert box.clip(0, 0, 20, 20) == box

    def test_normalized_maps_onto_the_unit_square(self):
        assert BBox(0, 0, 50, 100).normalized(100, 200) == BBox(0.0, 0.0, 0.5, 0.5)

    def test_normalized_rejects_a_zero_sized_page(self):
        with pytest.raises(ValueError, match="positive"):
            BBox(0, 0, 1, 1).normalized(0, 10)

    def test_pad_grows_in_every_direction(self):
        assert BBox(10, 10, 20, 20).pad(5) == BBox(5, 5, 25, 25)


class TestBBoxRelations:
    def test_union(self):
        assert BBox(0, 0, 10, 10).union(BBox(5, 5, 20, 20)) == BBox(0, 0, 20, 20)

    def test_union_of_many(self):
        boxes = [BBox(0, 0, 2, 2), BBox(10, 1, 12, 3), BBox(4, 4, 5, 5)]
        assert BBox.union_all(boxes) == BBox(0, 0, 12, 5)

    def test_union_all_rejects_empty_input(self):
        with pytest.raises(ValueError, match="at least one box"):
            BBox.union_all([])

    def test_intersection_of_overlapping_boxes(self):
        assert BBox(0, 0, 10, 10).intersection(BBox(5, 5, 20, 20)) == BBox(5, 5, 10, 10)

    def test_intersection_of_disjoint_boxes_is_none(self):
        assert BBox(0, 0, 5, 5).intersection(BBox(6, 6, 9, 9)) is None

    def test_touching_boxes_do_not_intersect(self):
        assert not BBox(0, 0, 5, 5).intersects(BBox(5, 0, 10, 5))

    def test_iou_of_identical_boxes_is_one(self):
        box = BBox(0, 0, 10, 10)
        assert box.iou(box) == pytest.approx(1.0)

    def test_iou_of_disjoint_boxes_is_zero(self):
        assert BBox(0, 0, 5, 5).iou(BBox(10, 10, 15, 15)) == 0.0

    def test_iou_of_half_overlap(self):
        assert BBox(0, 0, 10, 10).iou(BBox(5, 0, 15, 10)) == pytest.approx(1 / 3)

    def test_contains_a_nested_box(self):
        assert BBox(0, 0, 10, 10).contains(BBox(2, 2, 4, 4))
        assert not BBox(0, 0, 10, 10).contains(BBox(2, 2, 40, 4))


class TestBBoxSerialisation:
    def test_round_trips_through_a_list(self):
        box = BBox(1, 2, 3, 4)
        assert BBox.from_tuple(box.as_tuple()) == box

    def test_xywh_view(self):
        assert BBox(10, 20, 40, 60).as_xywh() == (10, 20, 30, 40)

    def test_as_tuple_is_plain_numbers(self):
        assert BBox(1, 2, 3, 4).as_tuple() == (1, 2, 3, 4)


class TestPolygon:
    def test_bbox_encloses_every_point(self):
        polygon = Polygon(((0, 0), (10, 2), (9, 8), (1, 6)))

        assert polygon.bbox == BBox(0, 0, 10, 8)

    def test_requires_at_least_three_points(self):
        with pytest.raises(ValueError, match="three points"):
            Polygon(((0, 0), (1, 1)))

    def test_translate_moves_every_point(self):
        polygon = Polygon(((0, 0), (2, 0), (2, 2)))

        assert polygon.translate(1, 1).points == ((1, 1), (3, 1), (3, 3))

    def test_from_bbox_produces_a_clockwise_quad(self):
        polygon = Polygon.from_bbox(BBox(0, 0, 10, 5))

        assert polygon.points == ((0, 0), (10, 0), (10, 5), (0, 5))

    def test_flat_serialisation_round_trips(self):
        polygon = Polygon(((0, 0), (2, 0), (2, 2)))

        assert Polygon.from_flat(polygon.as_flat()) == polygon

    def test_is_hashable_and_immutable(self):
        polygon = Polygon(((0, 0), (2, 0), (2, 2)))

        assert len({polygon, Polygon(((0, 0), (2, 0), (2, 2)))}) == 1
        with pytest.raises(FrozenInstanceError):
            polygon.points = ()
