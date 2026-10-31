# ai-toy-sweeper（このリポジトリでの Claude Code への指示）

- 共通の開発ルール（日本語、TDD、フォールバックではなく assert、設定値の根拠、uv）はユーザーの CLAUDE.md にある。
- 最終版の説明は
  README.md と doc/（レポート・手順書）、設定は config/、モデルは checkpoints/ppo_v36_s0_x2/。
- 3090 の PC（学習、Gazebo）と MacBook の二台が同じ main を使う。作業の前とプッシュの前に `git pull --rebase`。
- data/camera/、子どもの声の録音、話しかける画面のログ（dev_log/field_app/）、.claude/ は git に入れない。プロセスは PID で
  止める（pkill を使わない）。学習・サーバーの起動は言われてから（3090 の GPU は他のプロジェクトでも使う）。
