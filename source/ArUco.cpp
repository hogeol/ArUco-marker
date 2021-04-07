#include "ArUco.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
	PlaySound("c.wav", 0, SND_FILENAME | SND_ASYNC | SND_LOOP);
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);

	CameraInit();
	glutInitWindowSize(screen_width, screen_height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow("Make 3Dimage on camera");
	//메뉴 생성
	bool create = createMenu();
	if (create == 0)
		return -1;
	system("cls");
	//openGL 창 초기화
	openGL_cube_initiallize();
	//디스플레이 콜백 함수
	glutDisplayFunc(display);
	//키보드 콜백함수
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(anlgeSetting);
	//마우스 콜백 함수
	glutMouseFunc(mouse);
	//reshpae 콜백함수 (윈도우 크기 변경 시 호출)
	glutReshapeFunc(reshape);
	//타이머 콜백 함수
	glutTimerFunc(0, timer1, 0);

	glutMainLoop();

	return 0;
}