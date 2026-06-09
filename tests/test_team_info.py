import unittest

from scripts.apply_team_info import load_team_info, render_log, render_tex


class TeamInfoTests(unittest.TestCase):
    def test_team_has_exactly_four_members(self):
        data = load_team_info()
        self.assertEqual(len(data["members"]), 4)

    def test_generated_outputs_include_known_member(self):
        data = load_team_info()
        self.assertIn("Đoàn Danh Long", render_tex(data))
        self.assertIn("20237354", render_log(data))


if __name__ == "__main__":
    unittest.main()
